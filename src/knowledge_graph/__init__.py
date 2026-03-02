"""
Layer 2: Knowledge Graph (HIGH Semantic Space)

Network of nodes & edges representing domain knowledge.
Bridges ontology facts to queryable graph structures in Neo4j.

Example: [Person] → hasAddress → [Address]

Components:
- graph_model: Node, Edge, Property type definitions
- neo4j_knowledge_graph: Neo4j knowledge graph operations (Cypher)
- graph_agent: Agent that generates Cypher queries from semantic context
- graph_rag: GraphRAG - combines graph traversal with LLM reasoning
- embeddings: Graph node/edge embedding generation
"""

from .graph_model import (
    GraphNode,
    GraphEdge,
    GraphProperty,
    GraphSchema,
    GraphPath,
)
from .neo4j_knowledge_graph import KnowledgeGraph, KnowledgeGraphConfig
from .graph_agent import KnowledgeGraphAgent, KnowledgeGraphAgentConfig
from .graph_rag import GraphRAG, GraphRAGConfig
from .embeddings import GraphEmbeddingService, EmbeddingConfig

__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphProperty",
    "GraphSchema",
    "GraphPath",
    "KnowledgeGraph",
    "KnowledgeGraphConfig",
    "KnowledgeGraphAgent",
    "KnowledgeGraphAgentConfig",
    "GraphRAG",
    "GraphRAGConfig",
    "GraphEmbeddingService",
    "EmbeddingConfig",
]
