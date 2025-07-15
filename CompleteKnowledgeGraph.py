import spacy
import pandas as pd
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
import re
from neo4j import GraphDatabase
import numpy as np

# Load NER model
try:
    nlp = spacy.load("en_core_sci_sm")
    print("✅ Using biomedical model: en_core_sci_sm")
except:
    nlp = spacy.load("en_core_web_sm")
    print("⚠️ Using general model: en_core_web_sm")

class CompleteKnowledgeGraphBuilder:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Section importance weights
        self.section_weights = {
            'Abstract': 3.0, 'Conclusion': 2.5, 'Conclusions': 2.5,
            'Results': 2.0, 'Discussion': 1.5, 'Introduction': 1.0,
            'Methods': 0.8, 'References': 0.1
        }
    
    def close(self):
        self.driver.close()
    
    def extract_entities_from_chunks(self, df_chunks, batch_size=1000):
        """Extract all entities from chunks using NER"""
        print(f"🔍 Extracting entities from {len(df_chunks)} chunks...")
        
        all_entity_mentions = []
        entity_frequencies = Counter()
        entity_contexts = defaultdict(list)
        entity_label_counts = defaultdict(Counter)
        
        for batch_start in tqdm(range(0, len(df_chunks), batch_size), desc="Processing chunks"):
            batch = df_chunks[batch_start:batch_start + batch_size]
            
            for _, chunk in batch.iterrows():
                # Extract entities using spaCy
                doc = nlp(chunk['chunk_text'])
                
                for ent in doc.ents:
                    entity_text = ent.text.lower().strip()
                    
                    # Basic filtering
                    if (len(entity_text) >= 3 and 
                        entity_text.replace(' ', '').replace('-', '').isalpha()):
                        
                        # Store mention
                        mention = {
                            'entity': entity_text,
                            'label': ent.label_,
                            'paper_id': chunk['paper_id'],
                            'section': chunk['section_title'],
                            'chunk_id': chunk['chunk_id'],
                            'start_pos': ent.start_char,
                            'end_pos': ent.end_char
                        }
                        all_entity_mentions.append(mention)
                        
                        # Update counters
                        entity_frequencies[entity_text] += 1
                        entity_label_counts[entity_text][ent.label_] += 1
                        entity_contexts[entity_text].append({
                            'section': chunk['section_title'],
                            'paper_id': chunk['paper_id'],
                            'label': ent.label_
                        })
        
        print(f"✅ Extracted {len(all_entity_mentions)} entity mentions")
        print(f"✅ Found {len(entity_frequencies)} unique entities")
        
        return all_entity_mentions, entity_frequencies, entity_contexts, entity_label_counts
    
    def separate_mentions_by_consistency(self, all_entity_mentions, entity_label_counts, min_frequency=2):
        """Separate mentions into clean and inconsistent lists"""
        print("🧹 Separating mentions by label consistency...")
        
        # Get dominant label for each entity
        entity_dominant_labels = {}
        entity_stats = {}
        
        for entity, label_counts in entity_label_counts.items():
            total_mentions = sum(label_counts.values())
            
            # Skip entities with very low frequency
            if total_mentions < min_frequency:
                continue
                
            # Get the most frequent label
            dominant_label, dominant_count = label_counts.most_common(1)[0]
            dominant_percentage = dominant_count / total_mentions
            
            entity_dominant_labels[entity] = dominant_label
            entity_stats[entity] = {
                'total_mentions': total_mentions,
                'dominant_label': dominant_label,
                'dominant_count': dominant_count,
                'dominant_percentage': dominant_percentage,
                'label_distribution': dict(label_counts)
            }
        
        # Separate mentions into clean and inconsistent lists
        clean_mentions = []
        inconsistent_mentions = []
        low_frequency_mentions = []
        
        for mention in all_entity_mentions:
            entity = mention['entity']
            
            # Check if entity has enough frequency to have a dominant label
            if entity not in entity_dominant_labels:
                low_frequency_mentions.append(mention)
            # Check if mention matches dominant label
            elif mention['label'] == entity_dominant_labels[entity]:
                clean_mentions.append(mention)
            else:
                inconsistent_mentions.append(mention)
        
        print(f"✅ Clean mentions: {len(clean_mentions)}")
        print(f"⚠️ Inconsistent mentions: {len(inconsistent_mentions)}")
        print(f"📉 Low frequency mentions: {len(low_frequency_mentions)}")
        print(f"📊 Entities with dominant labels: {len(entity_dominant_labels)}")
        
        # Show some statistics
        print("\n📈 Dominant Label Distribution:")
        label_dist = Counter(stats['dominant_label'] for stats in entity_stats.values())
        for label, count in label_dist.most_common(10):
            print(f"   {label:15} {count:5} entities")
        
        # Show entities with best label consistency
        print("\n🎯 Most Consistent Entities (>95% dominant label):")
        consistent_entities = [(entity, stats) for entity, stats in entity_stats.items() 
                             if stats['dominant_percentage'] >= 0.95]
        consistent_entities.sort(key=lambda x: x[1]['total_mentions'], reverse=True)
        
        for entity, stats in consistent_entities[:10]:
            print(f"   {entity:20} {stats['dominant_percentage']:.1%} {stats['dominant_label']:12} ({stats['total_mentions']} mentions)")
        
        # Analyze inconsistent mentions
        print("\n🔍 Inconsistent Mention Analysis:")
        inconsistent_by_entity = defaultdict(list)
        for mention in inconsistent_mentions:
            inconsistent_by_entity[mention['entity']].append(mention)
        
        # Show top entities with most inconsistent mentions
        top_inconsistent = sorted(inconsistent_by_entity.items(), 
                                key=lambda x: len(x[1]), reverse=True)[:10]
        print("   Top entities with inconsistent mentions:")
        for entity, mentions in top_inconsistent:
            dominant = entity_dominant_labels.get(entity, 'UNKNOWN')
            inconsistent_labels = Counter(m['label'] for m in mentions)
            print(f"     {entity:15} dominant:{dominant:12} inconsistent:{dict(inconsistent_labels)}")
        
        return clean_mentions, inconsistent_mentions, low_frequency_mentions, entity_stats
    
    def process_inconsistent_mentions(self, inconsistent_mentions, entity_stats):
        """Process inconsistent mentions for potential recovery or analysis"""
        print("\n🔧 Processing inconsistent mentions...")
        
        if not inconsistent_mentions:
            print("   No inconsistent mentions to process")
            return []
        
        # Group inconsistent mentions by entity
        inconsistent_by_entity = defaultdict(list)
        for mention in inconsistent_mentions:
            inconsistent_by_entity[mention['entity']].append(mention)
        
        recovered_mentions = []
        
        for entity, mentions in inconsistent_by_entity.items():
            if entity not in entity_stats:
                continue
                
            entity_info = entity_stats[entity]
            dominant_label = entity_info['dominant_label']
            
            # Strategy 1: Recover mentions that are close to dominant but different casing/variation
            # (For now, we'll just collect them for analysis)
            
            # Strategy 2: If inconsistent mentions form a significant secondary pattern
            inconsistent_labels = Counter(m['label'] for m in mentions)
            most_common_inconsistent = inconsistent_labels.most_common(1)
            
            if most_common_inconsistent:
                secondary_label, secondary_count = most_common_inconsistent[0]
                
                # If secondary label appears frequently and is biomedical, might be worth keeping
                if (secondary_count >= 3 and 
                    secondary_label in ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY'] and
                    dominant_label not in ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY']):
                    
                    # Recover these as they might be valid biomedical mentions
                    for mention in mentions:
                        if mention['label'] == secondary_label:
                            mention['recovery_reason'] = 'secondary_biomedical'
                            recovered_mentions.append(mention)
        
        print(f"   💾 Recovered mentions: {len(recovered_mentions)}")
        
        # Show recovery statistics
        if recovered_mentions:
            recovery_reasons = Counter(m['recovery_reason'] for m in recovered_mentions)
            print("   Recovery breakdown:")
            for reason, count in recovery_reasons.items():
                print(f"     {reason}: {count}")
        
        return recovered_mentions
    
    def analyze_mention_quality(self, clean_mentions, inconsistent_mentions, low_frequency_mentions):
        """Analyze the quality and distribution of different mention types"""
        print("\n📊 MENTION QUALITY ANALYSIS")
        print("="*50)
        
        total_mentions = len(clean_mentions) + len(inconsistent_mentions) + len(low_frequency_mentions)
        
        print(f"Total mentions: {total_mentions:,}")
        print(f"Clean mentions: {len(clean_mentions):,} ({len(clean_mentions)/total_mentions*100:.1f}%)")
        print(f"Inconsistent mentions: {len(inconsistent_mentions):,} ({len(inconsistent_mentions)/total_mentions*100:.1f}%)")
        print(f"Low frequency mentions: {len(low_frequency_mentions):,} ({len(low_frequency_mentions)/total_mentions*100:.1f}%)")
        
        # Analyze label distribution in each category
        print("\n🏷️ Label Distribution by Category:")
        
        print("  Clean mentions:")
        clean_labels = Counter(m['label'] for m in clean_mentions)
        for label, count in clean_labels.most_common(10):
            print(f"    {label:15} {count:6,}")
        
        print("  Inconsistent mentions:")
        inconsistent_labels = Counter(m['label'] for m in inconsistent_mentions)
        for label, count in inconsistent_labels.most_common(10):
            print(f"    {label:15} {count:6,}")
        
        return {
            'total': total_mentions,
            'clean': len(clean_mentions),
            'inconsistent': len(inconsistent_mentions),
            'low_frequency': len(low_frequency_mentions),
            'clean_labels': dict(clean_labels),
            'inconsistent_labels': dict(inconsistent_labels)
        }
    
    def filter_entities_by_importance(self, entity_stats, min_mentions=3):
        """Filter entities based on importance criteria"""
        print("🎯 Filtering entities by importance...")
        
        filtered_entities = []
        
        for entity, stats in entity_stats.items():
            total_mentions = stats['total_mentions']
            dominant_label = stats['dominant_label']
            dominant_percentage = stats['dominant_percentage']
            
            keep_entity = False
            filter_reason = None
            
            # Rule 1: High frequency entities
            if total_mentions >= 15:
                keep_entity = True
                filter_reason = 'high_frequency'
            
            # Rule 2: Biomedical entities with decent frequency
            elif (dominant_label in ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY'] and 
                  total_mentions >= 3):
                keep_entity = True
                filter_reason = 'biomedical'
            
            # Rule 3: Important contextual entities
            elif (dominant_label in ['ORG', 'PERSON', 'GPE'] and 
                  total_mentions >= 5):
                keep_entity = True
                filter_reason = 'contextual'
            
            # Rule 4: Very consistent entities (any label, but highly consistent)
            elif (dominant_percentage >= 0.95 and 
                  total_mentions >= 5):
                keep_entity = True
                filter_reason = 'highly_consistent'
            
            if keep_entity:
                filtered_entities.append({
                    'entity': entity,
                    'frequency': total_mentions,
                    'dominant_label': dominant_label,
                    'dominant_percentage': dominant_percentage,
                    'label_distribution': stats['label_distribution'],
                    'filter_reason': filter_reason,
                    'importance_score': total_mentions * dominant_percentage
                })
        
        # Sort by importance
        filtered_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        
        print(f"✅ Filtered to {len(filtered_entities)} important entities")
        
        # Show distribution
        reasons = Counter(e['filter_reason'] for e in filtered_entities)
        print("\n📊 Filter Reason Distribution:")
        for reason, count in reasons.items():
            print(f"   {reason}: {count}")
        
        return filtered_entities
        """Filter entities based on importance criteria"""
        print("🎯 Filtering entities by importance...")
        
        filtered_entities = []
        
        for entity, stats in entity_stats.items():
            total_mentions = stats['total_mentions']
            dominant_label = stats['dominant_label']
            dominant_percentage = stats['dominant_percentage']
            
            keep_entity = False
            filter_reason = None
            
            # Rule 1: High frequency entities
            if total_mentions >= 15:
                keep_entity = True
                filter_reason = 'high_frequency'
            
            # Rule 2: Biomedical entities with decent frequency
            elif (dominant_label in ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY'] and 
                  total_mentions >= 3):
                keep_entity = True
                filter_reason = 'biomedical'
            
            # Rule 3: Important contextual entities
            elif (dominant_label in ['ORG', 'PERSON', 'GPE'] and 
                  total_mentions >= 5):
                keep_entity = True
                filter_reason = 'contextual'
            
            # Rule 4: Very consistent entities (any label, but highly consistent)
            elif (dominant_percentage >= 0.95 and 
                  total_mentions >= 5):
                keep_entity = True
                filter_reason = 'highly_consistent'
            
            if keep_entity:
                filtered_entities.append({
                    'entity': entity,
                    'frequency': total_mentions,
                    'dominant_label': dominant_label,
                    'dominant_percentage': dominant_percentage,
                    'label_distribution': stats['label_distribution'],
                    'filter_reason': filter_reason,
                    'importance_score': total_mentions * dominant_percentage
                })
        
        # Sort by importance
        filtered_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        
        print(f"✅ Filtered to {len(filtered_entities)} important entities")
        
        # Show distribution
        reasons = Counter(e['filter_reason'] for e in filtered_entities)
        print("\n📊 Filter Reason Distribution:")
        for reason, count in reasons.items():
            print(f"   {reason}: {count}")
        
        return filtered_entities
    
    def extract_entity_cooccurrences(self, clean_mentions, filtered_entities, min_strength=3):
        """Extract entity co-occurrence relationships"""
        print("🔗 Extracting entity co-occurrences...")
        
        # Get list of filtered entity names
        valid_entities = set(e['entity'] for e in filtered_entities)
        
        # Group mentions by chunk (only clean mentions)
        chunk_entities = defaultdict(set)
        for mention in clean_mentions:
            if mention['entity'] in valid_entities:
                chunk_entities[mention['chunk_id']].add(mention['entity'])
        
        # Calculate co-occurrences
        cooccurrence_counts = Counter()
        
        for chunk_id, entities in tqdm(chunk_entities.items(), desc="Computing co-occurrences"):
            entities_list = list(entities)
            for i, entity1 in enumerate(entities_list):
                for entity2 in entities_list[i+1:]:
                    # Ensure consistent ordering
                    pair = tuple(sorted([entity1, entity2]))
                    cooccurrence_counts[pair] += 1
        
        # Filter by minimum strength
        cooccurrences = []
        for (entity1, entity2), strength in cooccurrence_counts.items():
            if strength >= min_strength:
                cooccurrences.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'strength': strength,
                    'relationship_type': 'CO_OCCURS'
                })
        
        print(f"✅ Found {len(cooccurrences)} co-occurrence relationships")
        return cooccurrences
    
    def extract_paper_similarities(self, clean_mentions, filtered_entities, min_shared=5):
        """Extract paper similarity relationships"""
        print("📄 Computing paper similarities...")
        
        valid_entities = set(e['entity'] for e in filtered_entities)
        
        # Group entities by paper (only clean mentions)
        paper_entities = defaultdict(set)
        for mention in clean_mentions:
            if mention['entity'] in valid_entities:
                paper_entities[mention['paper_id']].add(mention['entity'])
        
        # Calculate similarities
        paper_ids = list(paper_entities.keys())
        similarities = []
        
        for i, paper1 in enumerate(tqdm(paper_ids, desc="Computing similarities")):
            for paper2 in paper_ids[i+1:]:
                entities1 = paper_entities[paper1]
                entities2 = paper_entities[paper2]
                
                shared = entities1 & entities2
                if len(shared) >= min_shared:
                    # Jaccard similarity
                    union = entities1 | entities2
                    jaccard = len(shared) / len(union) if union else 0
                    
                    similarities.append({
                        'paper1': paper1,
                        'paper2': paper2,
                        'shared_entities': len(shared),
                        'jaccard_similarity': jaccard,
                        'relationship_type': 'SIMILAR_TO'
                    })
        
        print(f"✅ Found {len(similarities)} paper similarity relationships")
        return similarities
    
    def create_neo4j_indexes(self):
        """Create indexes for better performance"""
        print("📊 Creating Neo4j indexes...")
        
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create indexes
            session.run("CREATE INDEX paper_id_idx IF NOT EXISTS FOR (p:Paper) ON (p.paper_id)")
            session.run("CREATE INDEX entity_text_idx IF NOT EXISTS FOR (e:Entity) ON (e.text)")
            session.run("CREATE INDEX entity_importance_idx IF NOT EXISTS FOR (e:Entity) ON (e.importance_score)")
        
        print("✅ Indexes created")
    
    def create_paper_nodes(self, df_chunks):
        """Create paper nodes in Neo4j"""
        print("📝 Creating paper nodes...")
        
        # Get unique papers with metadata
        papers = df_chunks.groupby('paper_id').agg({
            'paper_title': 'first',
            'chunk_id': 'count'
        }).reset_index()
        papers.columns = ['paper_id', 'title', 'chunk_count']
        
        with self.driver.session() as session:
            batch_size = 500
            for i in tqdm(range(0, len(papers), batch_size), desc="Creating papers"):
                batch = papers[i:i+batch_size]
                
                papers_data = []
                for _, paper in batch.iterrows():
                    papers_data.append({
                        'paper_id': paper['paper_id'],
                        'title': str(paper['title'])[:500],  # Limit title length
                        'chunk_count': int(paper['chunk_count'])
                    })
                
                session.run("""
                    UNWIND $papers AS paper
                    CREATE (p:Paper {
                        paper_id: paper.paper_id,
                        title: paper.title,
                        chunk_count: paper.chunk_count
                    })
                """, papers=papers_data)
        
        print(f"✅ Created {len(papers)} paper nodes")
    
    def create_entity_nodes(self, filtered_entities):
        """Create entity nodes in Neo4j"""
        print("🏷️ Creating entity nodes...")
        
        with self.driver.session() as session:
            batch_size = 500
            for i in tqdm(range(0, len(filtered_entities), batch_size), desc="Creating entities"):
                batch = filtered_entities[i:i+batch_size]
                
                entities_data = []
                for entity in batch:
                    entities_data.append({
                        'text': entity['entity'],
                        'frequency': entity['frequency'],
                        'dominant_label': entity['dominant_label'],
                        'dominant_percentage': entity['dominant_percentage'],
                        'filter_reason': entity['filter_reason'],
                        'importance_score': entity['importance_score']
                    })
                
                session.run("""
                    UNWIND $entities AS entity
                    CREATE (e:Entity {
                        text: entity.text,
                        frequency: entity.frequency,
                        dominant_label: entity.dominant_label,
                        dominant_percentage: entity.dominant_percentage,
                        filter_reason: entity.filter_reason,
                        importance_score: entity.importance_score
                    })
                """, entities=entities_data)
        
        print(f"✅ Created {len(filtered_entities)} entity nodes")
    
    def create_mentions_relationships(self, clean_mentions, filtered_entities):
        """Create MENTIONS relationships (Paper -> Entity) using only clean mentions"""
        print("🔗 Creating MENTIONS relationships...")
        
        valid_entities = set(e['entity'] for e in filtered_entities)
        
        # Filter clean mentions to only include valid entities
        valid_mentions = [m for m in clean_mentions if m['entity'] in valid_entities]
        
        with self.driver.session() as session:
            batch_size = 1000
            for i in tqdm(range(0, len(valid_mentions), batch_size), desc="Creating MENTIONS"):
                batch = valid_mentions[i:i+batch_size]
                
                mentions_data = []
                for mention in batch:
                    mentions_data.append({
                        'paper_id': mention['paper_id'],
                        'entity_text': mention['entity'],
                        'section': mention['section'],
                        'entity_label': mention['label']
                    })
                
                session.run("""
                    UNWIND $mentions AS mention
                    MATCH (p:Paper {paper_id: mention.paper_id})
                    MATCH (e:Entity {text: mention.entity_text})
                    CREATE (p)-[:MENTIONS {
                        section: mention.section,
                        entity_label: mention.entity_label
                    }]->(e)
                """, mentions=mentions_data)
        
        print(f"✅ Created {len(valid_mentions)} MENTIONS relationships")
    
    def create_cooccurrence_relationships(self, cooccurrences):
        """Create CO_OCCURS relationships (Entity <-> Entity)"""
        print("🔗 Creating CO_OCCURS relationships...")
        
        with self.driver.session() as session:
            batch_size = 500
            for i in tqdm(range(0, len(cooccurrences), batch_size), desc="Creating CO_OCCURS"):
                batch = cooccurrences[i:i+batch_size]
                
                cooccur_data = []
                for cooccur in batch:
                    cooccur_data.append({
                        'entity1': cooccur['entity1'],
                        'entity2': cooccur['entity2'],
                        'strength': cooccur['strength']
                    })
                
                session.run("""
                    UNWIND $cooccurs AS cooccur
                    MATCH (e1:Entity {text: cooccur.entity1})
                    MATCH (e2:Entity {text: cooccur.entity2})
                    CREATE (e1)-[:CO_OCCURS {
                        strength: cooccur.strength
                    }]->(e2)
                    CREATE (e2)-[:CO_OCCURS {
                        strength: cooccur.strength
                    }]->(e1)
                """, cooccurs=cooccur_data)
        
        print(f"✅ Created {len(cooccurrences) * 2} CO_OCCURS relationships (bidirectional)")
    
    def create_similarity_relationships(self, similarities):
        """Create SIMILAR_TO relationships (Paper <-> Paper)"""
        print("🔗 Creating SIMILAR_TO relationships...")
        
        with self.driver.session() as session:
            batch_size = 500
            for i in tqdm(range(0, len(similarities), batch_size), desc="Creating SIMILAR_TO"):
                batch = similarities[i:i+batch_size]
                
                similar_data = []
                for similar in batch:
                    similar_data.append({
                        'paper1': similar['paper1'],
                        'paper2': similar['paper2'],
                        'shared_entities': similar['shared_entities'],
                        'jaccard_similarity': similar['jaccard_similarity']
                    })
                
                session.run("""
                    UNWIND $similars AS similar
                    MATCH (p1:Paper {paper_id: similar.paper1})
                    MATCH (p2:Paper {paper_id: similar.paper2})
                    CREATE (p1)-[:SIMILAR_TO {
                        shared_entities: similar.shared_entities,
                        jaccard_similarity: similar.jaccard_similarity
                    }]->(p2)
                    CREATE (p2)-[:SIMILAR_TO {
                        shared_entities: similar.shared_entities,
                        jaccard_similarity: similar.jaccard_similarity
                    }]->(p1)
                """, similars=similar_data)
        
        print(f"✅ Created {len(similarities) * 2} SIMILAR_TO relationships (bidirectional)")
    
    def build_complete_knowledge_graph(self, df_chunks):
        """Complete pipeline to build knowledge graph"""
        print("="*80)
        print("🚀 BUILDING CLEAN KNOWLEDGE GRAPH (MENTION SEPARATION APPROACH)")
        print("="*80)
        
        # Step 1: Extract all entities and mentions
        print("\n🔍 STEP 1: ENTITY EXTRACTION")
        all_mentions, entity_frequencies, entity_contexts, entity_label_counts = self.extract_entities_from_chunks(df_chunks)
        
        # Step 2: Separate mentions into clean, inconsistent, and low-frequency
        print("\n🧹 STEP 2: SEPARATE MENTIONS BY CONSISTENCY")
        clean_mentions, inconsistent_mentions, low_frequency_mentions, entity_stats = self.separate_mentions_by_consistency(
            all_mentions, entity_label_counts, min_frequency=2
        )
        
        # Step 3: Analyze mention quality
        print("\n📊 STEP 3: MENTION QUALITY ANALYSIS")
        quality_stats = self.analyze_mention_quality(clean_mentions, inconsistent_mentions, low_frequency_mentions)
        
        # Step 4: Process inconsistent mentions for potential recovery
        print("\n🔧 STEP 4: PROCESS INCONSISTENT MENTIONS")
        recovered_mentions = self.process_inconsistent_mentions(inconsistent_mentions, entity_stats)
        
        # Combine clean mentions with recovered mentions
        all_clean_mentions = clean_mentions + recovered_mentions
        print(f"   Final clean mentions: {len(all_clean_mentions):,}")
        
        # Step 5: Filter entities by importance
        print("\n🎯 STEP 5: FILTER ENTITIES BY IMPORTANCE")
        filtered_entities = self.filter_entities_by_importance(entity_stats, min_mentions=3)
        
        # Step 6: Extract relationships from clean data
        print("\n🔗 STEP 6: RELATIONSHIP EXTRACTION (CLEAN DATA)")
        cooccurrences = self.extract_entity_cooccurrences(all_clean_mentions, filtered_entities)
        similarities = self.extract_paper_similarities(all_clean_mentions, filtered_entities)
        
        # Step 7: Create Neo4j graph
        print("\n🗄️ STEP 7: NEO4J GRAPH CREATION")
        self.create_neo4j_indexes()
        self.create_paper_nodes(df_chunks)
        self.create_entity_nodes(filtered_entities)
        self.create_mentions_relationships(all_clean_mentions, filtered_entities)
        self.create_cooccurrence_relationships(cooccurrences)
        self.create_similarity_relationships(similarities)
        
        # Step 8: Process remaining inconsistent mentions for additional analysis
        print("\n🔍 STEP 8: FINAL ANALYSIS OF REMAINING INCONSISTENT DATA")
        remaining_inconsistent = [m for m in inconsistent_mentions if m not in recovered_mentions]
        
        if remaining_inconsistent:
            print(f"   Remaining inconsistent mentions: {len(remaining_inconsistent):,}")
            
            # Analyze what we couldn't recover
            remaining_entities = Counter(m['entity'] for m in remaining_inconsistent)
            remaining_labels = Counter(m['label'] for m in remaining_inconsistent)
            
            print(f"   Top inconsistent entities:")
            for entity, count in remaining_entities.most_common(10):
                print(f"     {entity:15} {count:4} inconsistent mentions")
            
            print(f"   Inconsistent label distribution:")
            for label, count in remaining_labels.most_common(10):
                print(f"     {label:15} {count:4} mentions")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ KNOWLEDGE GRAPH WITH MENTION SEPARATION COMPLETE!")
        print("="*80)
        
        with self.driver.session() as session:
            # Count nodes
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) as count").single()['count']
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
            
            # Count relationships
            mentions_count = session.run("MATCH ()-[:MENTIONS]->() RETURN count(*) as count").single()['count']
            cooccurs_count = session.run("MATCH ()-[:CO_OCCURS]->() RETURN count(*) as count").single()['count']
            similar_count = session.run("MATCH ()-[:SIMILAR_TO]->() RETURN count(*) as count").single()['count']
            
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Papers: {paper_count:,}")
            print(f"   Entities: {entity_count:,}")
            print(f"   MENTIONS relationships: {mentions_count:,}")
            print(f"   CO_OCCURS relationships: {cooccurs_count:,}")
            print(f"   SIMILAR_TO relationships: {similar_count:,}")
            print(f"   Total relationships: {mentions_count + cooccurs_count + similar_count:,}")
        
        # Show mention processing summary
        print(f"\n📈 MENTION PROCESSING SUMMARY:")
        print(f"   Original mentions: {len(all_mentions):,}")
        print(f"   Clean mentions: {len(clean_mentions):,}")
        print(f"   Recovered mentions: {len(recovered_mentions):,}")
        print(f"   Final clean mentions: {len(all_clean_mentions):,}")
        print(f"   Inconsistent (not recovered): {len(remaining_inconsistent):,}")
        print(f"   Low frequency: {len(low_frequency_mentions):,}")
        print(f"   Data utilization: {(len(all_clean_mentions) / len(all_mentions)) * 100:.1f}%")
        
        # Show top entities by consistency
        print(f"\n🏆 TOP 10 MOST CONSISTENT ENTITIES:")
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity) 
                RETURN e.text, e.dominant_percentage, e.dominant_label, e.frequency
                ORDER BY e.dominant_percentage DESC, e.frequency DESC
                LIMIT 10
            """)
            for record in result:
                print(f"   {record['e.text']:20} {record['e.dominant_percentage']:.1%} {record['e.dominant_label']:12} ({record['e.frequency']} mentions)")
        
        return {
            'filtered_entities': filtered_entities,
            'cooccurrences': cooccurrences,
            'similarities': similarities,
            'clean_mentions': len(clean_mentions),
            'recovered_mentions': len(recovered_mentions),
            'inconsistent_mentions': len(remaining_inconsistent),
            'low_frequency_mentions': len(low_frequency_mentions),
            'original_mentions': len(all_mentions),
            'quality_stats': quality_stats,
            'statistics': {
                'papers': paper_count,
                'entities': entity_count,
                'mentions': mentions_count,
                'cooccurs': cooccurs_count,
                'similar': similar_count
            }
        }

# Main execution function
def main():
    """Main execution function"""
    # Neo4j connection
    NEO4J_URI = "neo4j+s://d0a53a4d.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "_fol0eGFfgI06cVj1Sk9d1-BkABbqEU6kRseBN7wWSc"
    
    # Initialize builder
    kg_builder = CompleteKnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Build complete knowledge graph
        results = kg_builder.build_complete_knowledge_graph(df_chunks)
        
        print("\n🎯 Sample queries for your clean knowledge graph:")
        print("""
        // Find most consistent entities (highest dominant_percentage)
        MATCH (e:Entity) 
        WHERE e.dominant_percentage >= 0.9
        RETURN e.text, e.dominant_percentage, e.dominant_label, e.frequency
        ORDER BY e.dominant_percentage DESC

        // Find biomedical entities only
        MATCH (e:Entity)
        WHERE e.dominant_label IN ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY']
        RETURN e.text, e.dominant_label, e.frequency
        ORDER BY e.frequency DESC

        // Find strong co-occurrences between biomedical entities
        MATCH (e1:Entity)-[r:CO_OCCURS]-(e2:Entity)
        WHERE e1.dominant_label IN ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY']
          AND e2.dominant_label IN ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY']
        RETURN e1.text, e2.text, r.strength
        ORDER BY r.strength DESC LIMIT 20

        // All mentions are now clean (match their entity's dominant label)
        MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
        WHERE m.entity_label = e.dominant_label
        RETURN count(*) as clean_mentions
        """)
        
        print(f"\n✨ Mention Processing Results:")
        print(f"   Original mentions: {results['original_mentions']:,}")
        print(f"   Clean mentions: {results['clean_mentions']:,}")
        print(f"   Recovered mentions: {results['recovered_mentions']:,}")
        print(f"   Inconsistent (unused): {results['inconsistent_mentions']:,}")
        print(f"   Low frequency (unused): {results['low_frequency_mentions']:,}")
        print(f"   Data utilization: {((results['clean_mentions'] + results['recovered_mentions']) / results['original_mentions']) * 100:.1f}%")
        
        return results
    
    finally:
        kg_builder.close()

# Run the complete pipeline
if __name__ == "__main__":
    results = main()
