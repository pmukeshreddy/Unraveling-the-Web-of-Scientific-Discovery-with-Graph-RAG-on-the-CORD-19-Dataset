import os
import json
import re
import pandas as pd
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
from neo4j import GraphDatabase
import time
from transformers import pipeline
from itertools import combinations

class KnowledgeGraphBuilder:
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # Load NER pipeline (general model, but good for starters)
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        print("✅ Loaded transformers NER model")

    def close(self):
        
        self.driver.close()
    def extract_entities_from_chunks(self, df_chunks, batch_size=500):
        
        print(f"🔍 Extracting entities from {len(df_chunks)} chunks...")
        all_mentions = []
        entity_frequencies = Counter()

        for batch_start in tqdm(range(0, len(df_chunks), batch_size), desc="Processing chunks"):
            batch = df_chunks[batch_start:batch_start + batch_size]
            for _, chunk in batch.iterrows():
                entities = self.ner_pipeline(chunk['chunk_text'])
                for ent in entities:
                    if ent['score'] > 0.7:  # Confidence threshold
                        entity_text = ent['word'].strip().lower()
                        label = ent['entity_group']
                        # Biomedical enhancement: If 'MISC' and matches bio patterns, tag as 'BIOMED'
                        if label == 'MISC' and re.search(r'(virus|cov|covid|disease|vaccine|gene|cell|protein|infection|patient|treatment)', entity_text):
                            label = 'BIOMED'
                        if len(entity_text) >= 3 and re.match(r'^[\w\s\-+&]+$', entity_text):  # Allow alphanum, hyphens
                            mention = {
                                'entity': entity_text,
                                'label': label,
                                'paper_id': chunk['paper_id'],
                                'section': chunk['section_title'],
                                'chunk_id': chunk['chunk_id'],
                                'start_pos': ent['start'],
                                'end_pos': ent['end']
                            }
                            all_mentions.append(mention)
                            entity_frequencies[entity_text] += 1

        print(f"✅ Extracted {len(all_mentions)} entity mentions")
        print(f"✅ Found {len(entity_frequencies)} unique entities")
        return all_mentions, entity_frequencies
    def filter_entities_by_importance(self, all_mentions, entity_frequencies, min_mentions=3, top_n=1000):
        print("🎯 Filtering entities by importance (enhanced version)...")
        
        # Section weights for contextual boost
        section_weights = {
            'Abstract': 3.0, 'Conclusions': 2.5, 'Results': 2.0,
            'Discussion': 1.5, 'Introduction': 1.0, 'Methods': 0.8,
            # Add more if needed; default to 1.0
        }
        
        # Precompute document frequency (DF) for TF-IDF like scoring
        entity_docs = defaultdict(set)  # Entities to set of paper_ids
        for mention in all_mentions:
            entity_docs[mention['entity']].add(mention['paper_id'])
        total_docs = len(set(m['paper_id'] for m in all_mentions))  # Unique papers
        
        filtered_entities = []
        for entity, freq in entity_frequencies.items():
            if freq < min_mentions:
                continue
            
            # Get mentions for this entity
            entity_mentions = [m for m in all_mentions if m['entity'] == entity]
            labels = [m['label'] for m in entity_mentions]
            dominant_label = Counter(labels).most_common(1)[0][0] if labels else 'UNKNOWN'
            
            # Weighted frequency: Sum section weights instead of raw freq
            weighted_freq = sum(section_weights.get(m['section'], 1.0) for m in entity_mentions)
            
            # TF-IDF like: Term Freq (weighted) * Inverse Doc Freq
            df = len(entity_docs[entity])
            idf = np.log(total_docs / (df + 1)) + 1  # Avoid div by zero
            tf_idf = weighted_freq * idf
            
            # Bio-boost: 1.5x for 'BIOMED', variable based on dominance
            label_boost = 1.5 if dominant_label == 'BIOMED' else 1.0
            dominance_ratio = Counter(labels)[dominant_label] / freq if freq > 0 else 1.0
            boost = label_boost * dominance_ratio  # E.g., if 80% BIOMED, boost accordingly
            
            importance_score = tf_idf * boost
            
            filtered_entities.append({
                'entity': entity,
                'frequency': freq,
                'dominant_label': dominant_label,
                'importance_score': importance_score
            })
        
        # Sort and cap to top_n
        filtered_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        filtered_entities = filtered_entities[:top_n]  # Prevent too many entities
        
        print(f"✅ Filtered to {len(filtered_entities)} important entities (top {top_n})")
        return filtered_entities
    def extract_entity_cooccurrences(self, all_mentions, filtered_entities, min_strength=2):
        print("🔗 Extracting entity co-occurrences...")
        valid_entities = set(e['entity'] for e in filtered_entities)
        chunk_entities = defaultdict(set)
        for mention in all_mentions:
            if mention['entity'] in valid_entities:
                chunk_entities[mention['chunk_id']].add(mention['entity'])
        
        cooccurrence_counts = Counter()
        for entities in tqdm(chunk_entities.values(), desc="Computing co-occurrences"):
            entities_list = list(entities)
            for pair in combinations(entities_list, 2):
                sorted_pair = tuple(sorted(pair))
                cooccurrence_counts[sorted_pair] += 1
        
        cooccurrences = [{'entity1': p[0], 'entity2': p[1], 'strength': s} for p, s in cooccurrence_counts.items() if s >= min_strength]
        print(f"✅ Found {len(cooccurrences)} co-occurrence relationships")
        return cooccurrences
    def extract_paper_similarities(self, all_mentions, filtered_entities, min_shared=3):
        print("📄 Computing paper similarities...")
        valid_entities = set(e['entity'] for e in filtered_entities)
        paper_entities = defaultdict(set)
        for mention in all_mentions:
            if mention['entity'] in valid_entities:
                paper_entities[mention['paper_id']].add(mention['entity'])
        
        paper_ids = list(paper_entities.keys())
        similarities = []
        for i in tqdm(range(len(paper_ids)), desc="Computing similarities"):
            paper1 = paper_ids[i]
            entities1 = paper_entities[paper1]
            if len(entities1) < min_shared:
                continue
            for j in range(i+1, len(paper_ids)):
                paper2 = paper_ids[j]
                entities2 = paper_entities[paper2]
                shared = len(entities1 & entities2)
                if shared >= min_shared:
                    union = len(entities1 | entities2)
                    jaccard = shared / union if union else 0
                    similarities.append({
                        'paper1': paper1,
                        'paper2': paper2,
                        'shared_entities': shared,
                        'jaccard_similarity': jaccard
                    })
        print(f"✅ Found {len(similarities)} paper similarity relationships")
        return similarities
    def create_neo4j_indexes(self):
        print("📊 Creating Neo4j indexes...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Clear DB
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.paper_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.text)")

    def create_paper_nodes(self, df_chunks):
        print("📝 Creating paper nodes...")
        papers = df_chunks[['paper_id', 'paper_title']].drop_duplicates().to_dict('records')
        with self.driver.session() as session:
            session.run("""
                UNWIND $papers AS paper
                CREATE (:Paper {paper_id: paper.paper_id, title: paper.paper_title})
            """, papers=papers)

    def create_entity_nodes(self, filtered_entities):
        print("🏷️ Creating entity nodes...")
        with self.driver.session() as session:
            session.run("""
                UNWIND $entities AS entity
                CREATE (:Entity {text: entity.entity, frequency: entity.frequency, dominant_label: entity.dominant_label})
            """, entities=filtered_entities)

    def create_mentions_relationships(self, all_mentions, filtered_entities):
        print("🔗 Creating MENTIONS relationships...")
        valid_entities = set(e['entity'] for e in filtered_entities)
        mentions = [m for m in all_mentions if m['entity'] in valid_entities]
        mentions_data = [{'paper_id': m['paper_id'], 'entity': m['entity'], 'section': m['section']} for m in mentions]
        with self.driver.session() as session:
            session.run("""
                UNWIND $mentions AS mention
                MATCH (p:Paper {paper_id: mention.paper_id})
                MATCH (e:Entity {text: mention.entity})
                CREATE (p)-[:MENTIONS {section: mention.section}]->(e)
            """, mentions=mentions_data)

    def create_cooccurrence_relationships(self, cooccurrences):
        print("🔗 Creating CO_OCCURS relationships...")
        with self.driver.session() as session:
            session.run("""
                UNWIND $cooccurs AS cooccur
                MATCH (e1:Entity {text: cooccur.entity1})
                MATCH (e2:Entity {text: cooccur.entity2})
                CREATE (e1)-[:CO_OCCURS {strength: cooccur.strength}]->(e2)
            """, cooccurs=cooccurrences)

    def create_similarity_relationships(self, similarities):
        print("🔗 Creating SIMILAR_TO relationships...")
        with self.driver.session() as session:
            session.run("""
                UNWIND $similars AS similar
                MATCH (p1:Paper {paper_id: similar.paper1})
                MATCH (p2:Paper {paper_id: similar.paper2})
                CREATE (p1)-[:SIMILAR_TO {jaccard: similar.jaccard_similarity}]->(p2)
            """, similars=similarities)

    def save_to_files(self, filtered_entities, cooccurrences, similarities, all_mentions):
        with open('entities.json', 'w') as f: json.dump(filtered_entities, f)
        with open('cooccurs.json', 'w') as f: json.dump(cooccurrences, f)
        with open('similarities.json', 'w') as f: json.dump(similarities, f)
        with open('mentions.json', 'w') as f: json.dump(all_mentions, f)
        print("💾 Saved to JSON files")
    def build_knowledge_graph(self, df_chunks):
        all_mentions, entity_frequencies = self.extract_entities_from_chunks(df_chunks)
        filtered_entities = self.filter_entities_by_importance(all_mentions, entity_frequencies)
        cooccurrences = self.extract_entity_cooccurrences(all_mentions, filtered_entities)
        similarities = self.extract_paper_similarities(all_mentions, filtered_entities)
        
        try:
            self.create_neo4j_indexes()
            self.create_paper_nodes(df_chunks)
            self.create_entity_nodes(filtered_entities)
            self.create_mentions_relationships(all_mentions, filtered_entities)
            self.create_cooccurrence_relationships(cooccurrences)
            self.create_similarity_relationships(similarities)
            print("✅ Graph built in Neo4j")
        except Exception as e:
            print(f"❌ Neo4j failed: {e}")
            self.save_to_files(filtered_entities, cooccurrences, similarities, all_mentions)
