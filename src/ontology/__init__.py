"""
Layer 1: Ontology / Fact-Based Model (HIGHEST Semantic Space)

Defines the domain through irreducible "elementary facts" using ORM (Object-Role Modeling).
Enables advanced AI reasoning over domain semantics.

Example fact: "The home address where <Person> is registered is <Address>"

Components:
- fact_model: OWL/RDF elementary fact definitions and ORM patterns
- ontology_store: Persistence layer for ontology facts (RDF graph store)
- ontology_agent: Agent that reasons over ontology facts using LLM
- reasoner: Fact-based inference engine for deriving new knowledge
"""

from .fact_model import (
    ElementaryFact,
    ObjectType,
    RoleType,
    Constraint,
    OntologyDefinition,
    FactPopulation,
)
from .ontology_store import OntologyStore, OntologyStoreConfig
from .ontology_agent import OntologyAgent, OntologyAgentConfig
from .reasoner import FactReasoner, InferenceResult

__all__ = [
    "ElementaryFact",
    "ObjectType",
    "RoleType",
    "Constraint",
    "OntologyDefinition",
    "FactPopulation",
    "OntologyStore",
    "OntologyStoreConfig",
    "OntologyAgent",
    "OntologyAgentConfig",
    "FactReasoner",
    "InferenceResult",
]
