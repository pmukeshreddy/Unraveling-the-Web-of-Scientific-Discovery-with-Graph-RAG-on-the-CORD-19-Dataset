# Unraveling-the-Web-of-Scientific-Discovery-with-Graph-RAG-on-the-CORD-19-Dataset


https://scientific-qa-app-914546417586.us-central1.run.app/ it's deployed here
# GraphRAG: Hybrid Retrieval for Academic Literature

![Neo4j](https://img.shields.io/badge/Neo4j-5.x-blue) ![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Store-red) ![Python](https://img.shields.io/badge/Python-3.9+-yellow) ![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview

This project implements a **Hybrid Retrieval Augmented Generation (RAG)** system designed to perform complex, multi-hop reasoning over a massive corpus of academic literature.

By combining the semantic search capabilities of a vector store (**Qdrant**) with the structural context of a Knowledge Graph (**Neo4j**), this system overcomes the limitations of standard vector-only RAG. It enables high-precision querying across **400,000+ ingested papers**, connecting disparate concepts, authors, and citations with sub-second latency.

## 🚀 Key Features

* **Hybrid Retrieval Engine:** Merges vector similarity search with graph traversal to find context that pure semantic search misses.
* **Massive Scale:** Ingested **400K+ papers**, resulting in a knowledge graph with **1.2M nodes** and **8M relationships**.
* **Multi-Hop Querying:** Capable of answering complex questions (e.g., *"What other concepts has the author of Paper X worked on regarding Topic Y?"*).
* **Optimized Performance:** Custom-tuned Cypher queries ensure sub-second retrieval times even for complex traversals.
* **Superior Relevance:** Internal benchmarks demonstrate **+25% higher relevance** compared to fine-tuned small language models (SLMs).
* **Standardized Quality:** Utilizes GPT-4 APIs for the generation layer to ensure consistent reasoning and low latency.

## 🏗️ Architecture & Schema

### 1. The Knowledge Graph (Neo4j)
The graph schema models the academic landscape to facilitate deep traversal.

* **Nodes:**
    * `Paper`: The core unit of the graph.
    * `Author`: Linked to papers they wrote.
    * `Concept`: Extracted keywords and entities (e.g., "Machine Learning," "CRISPR").
* **Relationships:**
    * `(Author)-[:AUTHORED]->(Paper)`
    * `(Paper)-[:CITES]->(Paper)`
    * `(Paper)-[:MENTIONS]->(Concept)`

### 2. The Vector Store (Qdrant)
* Stores high-dimensional embeddings of paper abstracts and full text.
* Used for the initial "fuzzy" retrieval to identify entry points into the graph before traversal begins.

## 📊 Performance

We compared this GraphRAG implementation against fine-tuned Small Language Models (SLMs) such as Llama-2-7b and Mistral.

| Metric | GraphRAG (Neo4j + Qdrant) | Fine-Tuned SLM | Improvement |
| :--- | :--- | :--- | :--- |
| **Relevance** | High (Context-aware) | Moderate (Hallucination prone) | **+25%** |
| **Latency** | Sub-second (Retrieval) | Variable | Comparable |
| **Multi-hop Accuracy** | High | Low | Significant |

