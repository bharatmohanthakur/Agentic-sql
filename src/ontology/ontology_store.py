"""
Ontology Store - Persistence layer for ontology definitions and fact populations.

Stores ontologies as graph structures in Neo4j, enabling:
- SPARQL-like graph queries over domain ontologies
- Semantic search for relevant facts
- Version-controlled ontology evolution
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .fact_model import (
    Constraint,
    ElementaryFact,
    FactInstance,
    FactPopulation,
    ObjectType,
    OntologyDefinition,
    RoleType,
)

logger = logging.getLogger(__name__)


@dataclass
class OntologyStoreConfig:
    """Configuration for the ontology store"""
    # Neo4j connection for graph storage
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Namespace for multi-tenancy
    namespace: str = "default"

    # Cache settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


class OntologyStore:
    """
    Persistence layer for ontology definitions.

    Uses Neo4j to store ontology elements as a graph:
    - ObjectType nodes with type hierarchy edges
    - ElementaryFact nodes connected to role/object type nodes
    - Constraint nodes attached to roles/facts
    - FactInstance nodes for population data

    Graph Structure:
        (ObjectType)-[:HAS_SUPERTYPE]->(ObjectType)
        (ElementaryFact)-[:HAS_ROLE]->(RoleType)-[:PLAYS]->(ObjectType)
        (Constraint)-[:CONSTRAINS]->(RoleType)
        (FactInstance)-[:INSTANCE_OF]->(ElementaryFact)
    """

    def __init__(self, config: OntologyStoreConfig):
        self.config = config
        self._driver = None
        self._cache: Dict[str, OntologyDefinition] = {}

    async def connect(self) -> None:
        """Initialize Neo4j connection and create indexes"""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
            )

            async with self._driver.session(database=self.config.neo4j_database) as session:
                await session.run("RETURN 1")

                # Create indexes for ontology elements
                for label in ["ObjectType", "ElementaryFact", "RoleType", "Constraint", "FactInstance"]:
                    await session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)"
                    )
                    await session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
                    )
                await session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (n:Ontology) ON (n.namespace)"
                )

            logger.info(f"Ontology store connected to Neo4j at {self.config.neo4j_uri}")

        except ImportError:
            raise ImportError("neo4j required: pip install neo4j")

    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            await self._driver.close()

    async def save_ontology(self, ontology: OntologyDefinition) -> bool:
        """
        Persist a complete ontology definition to Neo4j.

        Creates/updates all nodes and relationships.
        """
        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                # Create ontology root node
                await session.run(
                    """
                    MERGE (o:Ontology {id: $id})
                    SET o.name = $name, o.version = $version,
                        o.description = $description, o.namespace = $namespace,
                        o.updated_at = $updated_at
                    """,
                    id=ontology.id,
                    name=ontology.name,
                    version=ontology.version,
                    description=ontology.description,
                    namespace=ontology.namespace or self.config.namespace,
                    updated_at=datetime.utcnow().isoformat(),
                )

                # Create object types
                for ot in ontology.object_types:
                    await session.run(
                        """
                        MERGE (t:ObjectType {id: $id})
                        SET t.name = $name, t.kind = $kind,
                            t.description = $description,
                            t.reference_mode = $ref_mode,
                            t.data_type = $data_type
                        WITH t
                        MATCH (o:Ontology {id: $ont_id})
                        MERGE (o)-[:DEFINES_TYPE]->(t)
                        """,
                        id=ot.id, name=ot.name, kind=ot.kind.value,
                        description=ot.description, ref_mode=ot.reference_mode,
                        data_type=ot.data_type, ont_id=ontology.id,
                    )

                    # Create supertype relationship
                    if ot.supertype:
                        await session.run(
                            """
                            MATCH (sub:ObjectType {name: $sub_name})
                            MATCH (sup:ObjectType {name: $sup_name})
                            MERGE (sub)-[:HAS_SUPERTYPE]->(sup)
                            """,
                            sub_name=ot.name, sup_name=ot.supertype,
                        )

                # Create fact types with roles
                for ft in ontology.fact_types:
                    await session.run(
                        """
                        MERGE (f:ElementaryFact {id: $id})
                        SET f.name = $name, f.reading = $reading,
                            f.inverse_reading = $inverse_reading,
                            f.arity = $arity, f.is_derived = $is_derived,
                            f.derivation_rule = $derivation_rule
                        WITH f
                        MATCH (o:Ontology {id: $ont_id})
                        MERGE (o)-[:DEFINES_FACT]->(f)
                        """,
                        id=ft.id, name=ft.name, reading=ft.reading,
                        inverse_reading=ft.inverse_reading, arity=ft.arity,
                        is_derived=ft.is_derived, derivation_rule=ft.derivation_rule,
                        ont_id=ontology.id,
                    )

                    # Create roles and connect to object types
                    for i, role in enumerate(ft.roles):
                        await session.run(
                            """
                            MERGE (r:RoleType {id: $id})
                            SET r.name = $name, r.is_mandatory = $is_mandatory,
                                r.position = $position
                            WITH r
                            MATCH (f:ElementaryFact {id: $fact_id})
                            MERGE (f)-[:HAS_ROLE]->(r)
                            WITH r
                            MATCH (t:ObjectType {name: $type_name})
                            MERGE (r)-[:PLAYED_BY]->(t)
                            """,
                            id=role.id, name=role.name,
                            is_mandatory=role.is_mandatory, position=i,
                            fact_id=ft.id, type_name=role.object_type,
                        )

                # Create constraints
                for constraint in ontology.constraints:
                    await session.run(
                        """
                        MERGE (c:Constraint {id: $id})
                        SET c.kind = $kind, c.name = $name,
                            c.description = $description,
                            c.parameters = $parameters
                        """,
                        id=constraint.id, kind=constraint.kind.value,
                        name=constraint.name, description=constraint.description,
                        parameters=json.dumps(constraint.parameters),
                    )

                    for role_id in constraint.roles:
                        await session.run(
                            """
                            MATCH (c:Constraint {id: $c_id})
                            MATCH (r:RoleType {id: $r_id})
                            MERGE (c)-[:CONSTRAINS]->(r)
                            """,
                            c_id=constraint.id, r_id=role_id,
                        )

            # Update cache
            if self.config.enable_cache:
                self._cache[ontology.id] = ontology

            logger.info(f"Saved ontology '{ontology.name}' v{ontology.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to save ontology: {e}")
            return False

    async def load_ontology(self, ontology_id: str) -> Optional[OntologyDefinition]:
        """Load ontology definition from Neo4j"""
        # Check cache
        if self.config.enable_cache and ontology_id in self._cache:
            return self._cache[ontology_id]

        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                # Load ontology root
                result = await session.run(
                    "MATCH (o:Ontology {id: $id}) RETURN o",
                    id=ontology_id,
                )
                record = await result.single()
                if not record:
                    return None

                o = record["o"]

                # Load object types
                types_result = await session.run(
                    """
                    MATCH (o:Ontology {id: $id})-[:DEFINES_TYPE]->(t:ObjectType)
                    OPTIONAL MATCH (t)-[:HAS_SUPERTYPE]->(sup:ObjectType)
                    RETURN t, sup.name as supertype
                    """,
                    id=ontology_id,
                )
                object_types = []
                async for rec in types_result:
                    t = rec["t"]
                    object_types.append(ObjectType(
                        id=t["id"], name=t["name"],
                        kind=t.get("kind", "entity"),
                        description=t.get("description", ""),
                        reference_mode=t.get("reference_mode"),
                        supertype=rec.get("supertype"),
                        data_type=t.get("data_type"),
                    ))

                # Load fact types with roles
                facts_result = await session.run(
                    """
                    MATCH (o:Ontology {id: $id})-[:DEFINES_FACT]->(f:ElementaryFact)
                    OPTIONAL MATCH (f)-[:HAS_ROLE]->(r:RoleType)-[:PLAYED_BY]->(t:ObjectType)
                    RETURN f, collect({role: r, type_name: t.name}) as roles
                    ORDER BY f.name
                    """,
                    id=ontology_id,
                )
                fact_types = []
                async for rec in facts_result:
                    f = rec["f"]
                    roles = []
                    for role_data in rec["roles"]:
                        if role_data["role"]:
                            r = role_data["role"]
                            roles.append(RoleType(
                                id=r["id"], name=r["name"],
                                object_type=role_data["type_name"],
                                is_mandatory=r.get("is_mandatory", False),
                            ))
                    # Sort roles by position
                    roles.sort(key=lambda r: r.id)

                    fact_types.append(ElementaryFact(
                        id=f["id"], name=f["name"],
                        reading=f.get("reading", ""),
                        inverse_reading=f.get("inverse_reading"),
                        roles=roles,
                        is_derived=f.get("is_derived", False),
                        derivation_rule=f.get("derivation_rule"),
                    ))

                ontology = OntologyDefinition(
                    id=o["id"], name=o["name"],
                    version=o.get("version", "1.0.0"),
                    description=o.get("description", ""),
                    namespace=o.get("namespace", ""),
                    object_types=object_types,
                    fact_types=fact_types,
                )

                if self.config.enable_cache:
                    self._cache[ontology_id] = ontology

                return ontology

        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            return None

    async def find_facts_for_question(
        self,
        question: str,
        ontology_id: str,
        limit: int = 10,
    ) -> List[ElementaryFact]:
        """
        Find relevant elementary facts for a natural language question.

        Uses text matching on fact readings and object type names.
        """
        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                result = await session.run(
                    """
                    MATCH (o:Ontology {id: $ont_id})-[:DEFINES_FACT]->(f:ElementaryFact)
                    MATCH (f)-[:HAS_ROLE]->(r:RoleType)-[:PLAYED_BY]->(t:ObjectType)
                    WHERE f.reading CONTAINS $query
                       OR f.name CONTAINS $query
                       OR t.name CONTAINS $query
                    RETURN DISTINCT f, collect({role: r, type_name: t.name}) as roles
                    LIMIT $limit
                    """,
                    ont_id=ontology_id, query=question, limit=limit,
                )

                facts = []
                async for rec in result:
                    f = rec["f"]
                    roles = []
                    for role_data in rec["roles"]:
                        if role_data["role"]:
                            r = role_data["role"]
                            roles.append(RoleType(
                                id=r["id"], name=r["name"],
                                object_type=role_data["type_name"],
                            ))
                    facts.append(ElementaryFact(
                        id=f["id"], name=f["name"],
                        reading=f.get("reading", ""),
                        roles=roles,
                    ))

                return facts

        except Exception as e:
            logger.error(f"Failed to find facts: {e}")
            return []

    async def save_fact_population(self, population: FactPopulation) -> bool:
        """Store fact instances (population data) in Neo4j"""
        try:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                for instance in population.instances:
                    await session.run(
                        """
                        MERGE (fi:FactInstance {id: $id})
                        SET fi.fact_type = $fact_type,
                            fi.values = $values,
                            fi.source = $source,
                            fi.confidence = $confidence,
                            fi.timestamp = $timestamp
                        WITH fi
                        MATCH (f:ElementaryFact {name: $fact_type})
                        MERGE (fi)-[:INSTANCE_OF]->(f)
                        """,
                        id=instance.id,
                        fact_type=instance.fact_type,
                        values=json.dumps(instance.values),
                        source=instance.source,
                        confidence=instance.confidence,
                        timestamp=instance.timestamp.isoformat(),
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to save fact population: {e}")
            return False

    async def get_ontology_summary(self, ontology_id: str) -> Dict[str, Any]:
        """Get a summary of an ontology for LLM context"""
        ontology = await self.load_ontology(ontology_id)
        if not ontology:
            return {}

        return {
            "name": ontology.name,
            "version": ontology.version,
            "object_types": [
                {"name": ot.name, "kind": ot.kind.value, "description": ot.description}
                for ot in ontology.object_types
            ],
            "fact_types": [
                {"name": ft.name, "reading": ft.reading, "arity": ft.arity}
                for ft in ontology.fact_types
            ],
            "total_types": len(ontology.object_types),
            "total_facts": len(ontology.fact_types),
            "total_constraints": len(ontology.constraints),
        }
