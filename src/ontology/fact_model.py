"""
Fact-Based Model - ORM (Object-Role Modeling) Implementation

Implements the HIGHEST semantic space layer where domains are defined
through irreducible "elementary facts" rather than tables/columns.

Key Concepts:
- ObjectType: Named entity types (Person, Address, Order)
- RoleType: Relationships objects play in facts
- ElementaryFact: Irreducible fact connecting object types through roles
- Constraint: Business rules (uniqueness, mandatory, subset, etc.)
- FactPopulation: Instance data conforming to fact types

Example:
    fact = ElementaryFact(
        name="PersonHasAddress",
        reading="The home address where <Person> is registered is <Address>",
        roles=[
            RoleType(name="registered_person", object_type="Person"),
            RoleType(name="home_address", object_type="Address"),
        ],
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValueTypeKind(str, Enum):
    """Classification of value types in ORM"""
    ENTITY = "entity"          # Independently identifiable (Person, Order)
    VALUE = "value"            # Dependent on entity (Name, Date, Amount)
    OBJECTIFIED = "objectified"  # Fact type treated as object type


class ConstraintKind(str, Enum):
    """Types of ORM constraints"""
    UNIQUENESS = "uniqueness"           # Internal uniqueness constraint (IUC)
    MANDATORY = "mandatory"             # Mandatory role constraint
    FREQUENCY = "frequency"             # Min/max frequency
    RING = "ring"                       # Irreflexive, asymmetric, etc.
    SUBSET = "subset"                   # Role subset constraint
    EXCLUSION = "exclusion"             # Mutual exclusion
    EQUALITY = "equality"               # Role equality
    VALUE = "value_constraint"          # Allowed values
    EXTERNAL_UNIQUENESS = "external_uniqueness"  # Spans multiple fact types


class ObjectType(BaseModel):
    """
    An object type in the ontology - either an entity or a value type.

    Entity types are independently identifiable (Person, Order).
    Value types are dependent (Name, Date, Amount).
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    kind: ValueTypeKind = ValueTypeKind.ENTITY
    description: str = ""
    reference_mode: Optional[str] = None  # How entities are identified (e.g., "PersonId", "Name")
    supertype: Optional[str] = None       # Parent type for subtyping
    subtypes: List[str] = Field(default_factory=list)
    data_type: Optional[str] = None       # For value types: string, int, date, etc.
    allowed_values: Optional[List[str]] = None  # Enumerated value constraint
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_entity(self) -> bool:
        return self.kind == ValueTypeKind.ENTITY

    def is_value(self) -> bool:
        return self.kind == ValueTypeKind.VALUE


class RoleType(BaseModel):
    """
    A role that an object type plays in an elementary fact.

    Each role connects an object type to a fact type.
    Example: In "Person has Address", Person plays "has_address" role.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    object_type: str       # Reference to ObjectType.name
    is_mandatory: bool = False
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Constraint(BaseModel):
    """
    A constraint on fact types or roles.

    Constraints express business rules that govern the domain.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: ConstraintKind
    name: str = ""
    description: str = ""
    roles: List[str] = Field(default_factory=list)  # Role IDs this constraint applies to
    fact_types: List[str] = Field(default_factory=list)  # Fact type names
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Kind-specific params
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ElementaryFact(BaseModel):
    """
    An elementary fact type - the irreducible unit of domain knowledge.

    Each fact type defines a relationship between object types through roles.
    Facts are expressed in natural language readings for human understanding.

    Example:
        name: "PersonRegisteredAtAddress"
        reading: "The home address where <Person> is registered is <Address>"
        inverse_reading: "<Address> is the registered home address of <Person>"
        roles: [
            RoleType(name="registered_person", object_type="Person"),
            RoleType(name="home_address", object_type="Address"),
        ]
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    reading: str                          # Natural language reading
    inverse_reading: Optional[str] = None  # Inverse reading direction
    roles: List[RoleType] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    is_derived: bool = False               # Whether this fact is derivable from others
    derivation_rule: Optional[str] = None  # Derivation expression
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def arity(self) -> int:
        """Number of roles (unary, binary, ternary, etc.)"""
        return len(self.roles)

    @property
    def object_types(self) -> List[str]:
        """All object types participating in this fact"""
        return [role.object_type for role in self.roles]

    def get_role_for_type(self, object_type: str) -> Optional[RoleType]:
        """Get the role played by a specific object type"""
        for role in self.roles:
            if role.object_type == object_type:
                return role
        return None

    def to_triple(self) -> Optional[tuple]:
        """Convert binary fact to (subject, predicate, object) triple"""
        if self.arity == 2:
            return (
                self.roles[0].object_type,
                self.name,
                self.roles[1].object_type,
            )
        return None


class FactInstance(BaseModel):
    """
    A single instance of an elementary fact (a row of fact population).

    Example for "PersonRegisteredAtAddress":
        values: {"registered_person": "John Smith", "home_address": "123 Main St"}
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    fact_type: str  # Reference to ElementaryFact.name
    values: Dict[str, Any] = Field(default_factory=dict)  # role_name -> value
    source: Optional[str] = None  # Where this fact came from
    confidence: float = 1.0       # Confidence in this fact (0-1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FactPopulation(BaseModel):
    """
    A collection of fact instances for a fact type.

    Represents the "population" of a fact type - all known instances.
    """
    fact_type: str
    instances: List[FactInstance] = Field(default_factory=list)

    def add(self, values: Dict[str, Any], source: Optional[str] = None) -> FactInstance:
        """Add a new fact instance"""
        instance = FactInstance(fact_type=self.fact_type, values=values, source=source)
        self.instances.append(instance)
        return instance

    @property
    def count(self) -> int:
        return len(self.instances)


class OntologyDefinition(BaseModel):
    """
    Complete ontology definition for a domain.

    Contains all object types, fact types, and constraints that define
    the domain at the highest semantic level.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    version: str = "1.0.0"
    description: str = ""
    namespace: str = ""

    # Domain elements
    object_types: List[ObjectType] = Field(default_factory=list)
    fact_types: List[ElementaryFact] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_object_type(self, name: str) -> Optional[ObjectType]:
        """Look up an object type by name"""
        for ot in self.object_types:
            if ot.name == name:
                return ot
        return None

    def get_fact_type(self, name: str) -> Optional[ElementaryFact]:
        """Look up a fact type by name"""
        for ft in self.fact_types:
            if ft.name == name:
                return ft
        return None

    def get_facts_for_type(self, object_type: str) -> List[ElementaryFact]:
        """Get all fact types that involve a given object type"""
        return [
            ft for ft in self.fact_types
            if object_type in ft.object_types
        ]

    def get_related_types(self, object_type: str) -> Set[str]:
        """Get all object types related to a given type through facts"""
        related = set()
        for ft in self.get_facts_for_type(object_type):
            for ot in ft.object_types:
                if ot != object_type:
                    related.add(ot)
        return related

    def to_rdf_triples(self) -> List[tuple]:
        """Export ontology as RDF-style triples"""
        triples = []
        for ft in self.fact_types:
            triple = ft.to_triple()
            if triple:
                triples.append(triple)
        return triples

    def validate(self) -> List[str]:
        """Validate ontology consistency"""
        errors = []
        known_types = {ot.name for ot in self.object_types}

        for ft in self.fact_types:
            for role in ft.roles:
                if role.object_type not in known_types:
                    errors.append(
                        f"Fact '{ft.name}' references unknown type '{role.object_type}'"
                    )

        for ot in self.object_types:
            if ot.supertype and ot.supertype not in known_types:
                errors.append(
                    f"Type '{ot.name}' has unknown supertype '{ot.supertype}'"
                )

        return errors
