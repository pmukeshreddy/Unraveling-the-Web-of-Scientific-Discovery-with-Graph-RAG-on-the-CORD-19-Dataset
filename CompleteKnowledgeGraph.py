class CompleteKnowledgeGraph:
    def __init__(self,neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        
        # Section importance weights
        self.section_weights = {
            'Abstract': 3.0, 'Conclusion': 2.5, 'Conclusions': 2.5,
            'Results': 2.0, 'Discussion': 1.5, 'Introduction': 1.0,
            'Methods': 0.8, 'References': 0.1
        }
        self.label_statistics = {}
        self.important_labels = set()
    def close(self):
        self.driver.close()

    def extract_entities_from_chunks(self,df_chunks,batch_size=1000):
        all_entity_mentions = []
        entity_frequencies = Counter()
        entity_contexts = defaultdict(list)
        entity_label_counts = defaultdict(Counter)
        global_label_counts = Counter()

        for batch_start in tqdm(range(0, len(df_chunks), batch_size), desc="Processing chunks"):
            batch = df_chunks[batch_start:batch_start + batch_size]

            for _,chunk in batch.iterrows():
                doc = nlp(chunk["chunk_text"])

                for ent in doc.ents:
                    entity_text = ent.txt.lower().strip()

                    if (len(entity_text)) >=3 and entity_text.replace(' ', '').replace('-', '').isalpha()):
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
                        
                        entity_frequencies[entity_text] += 1
                        entity_label_counts[entity_text][ent.lable_] = +1
                        entity_contexts[entity_text].append({
                            'section': chunk['section_title'],
                            'paper_id': chunk['paper_id'] ,
                            "label" : ent.label_
                        })

        print(f"✅ Extracted {len(all_entity_mentions)} entity mentions")
        print(f"✅ Found {len(entity_frequencies)} unique entities")
        
        return all_entity_mentions, entity_frequencies, entity_contexts ,entity_label_counts 

    
    def analyze_label_importance(self,global_label_counts,entity_label_counts,top_k=10):
        total_mentions = sum(global_label_counts.values())
        label_stats = {}
        for label,count in global_label_counts.items():
            entities_with_label = sum(1 for entity_labels in entity_label_counts.values() if label in entity_labels)
            avg_freq = count/entities_with_label if entities_with_label > 0 else 0
            label_stats[label] = {"total_mentions":count,"percentage":(count/total_mentions)*100,"unique_entities":entities_with_label,
                                 "avg_frequency":avg_freq,"importance_score":count*entities_with_label}
        sorted_labels = sorted(label_stats.items(),key=lambda x:x[1]["importance_score"],reverse=True)

        important_labels = set()
        for i , (label,stats) in enumerate(sorted_labels[:top_k]):
            important_labels.add(label)

        self.label_statistics = label_stats
        self.important_labels = important_labels

        return label_stats , important_labels

    def get_entity_type_distribution(self,entity_label_counts,entity,important_labels):
        label_counts = entity_label_counts[entity]
        total_mentions = sum(label_counts.values())

        important_count = sum(count for label,count in label_counts.items() if label in important_labels)

        other_count = total_mentions - important_count

        important_pct = important_count/total_mentions if total_mentions >0 else 0
        other_pct = other_count / total_mentions if total_mentions>0 else 0

        most_frequent_labels = label_counts.most_common(1)[0][0] if label_counts else "UNKNOWN"
        
        
    def filter_entities_dynamically(self,all_entity_mentions, entity_frequencies, entity_contexts):
        filtered_enties = []

        for entity , frequency in entity_frequencies.items():
            contexts = entity_contexts[entity]

            paper_counts = len(set(ctx['paper_id'] for ctx in contexts))
            section_diversity = len(set(ctx['section'] for ctx in contexts))

            weighted_score = sum(self.section_weights.get(ctx['section'], 1.0) 
                               for ctx in contexts)
            entity_labels = set(mention['label'] for mention in all_entity_mentions 
                              if mention['entity'] == entity)
            

            keep_entity = False
            filter_reason = None

            if any(label in ['DISEASE', 'CHEMICAL', 'GENE', 'DRUG', 'ANATOMY'] for label in entity_labels):
                if frequency >= 2:
                    keep_enity = True
                    filter_reason = "biomedical_entity"
            elif any(indicator in entity for indicator in self.medical_indicators):
                if frequency >= 2:
                    keep_enity = True
                    filter_reason = 'medical_context'
            elif paper_count >= 3:
                if frequency >= 4:
                    keep_enity = True
                    filter_reason = 'section_diversity'
            elif weighted_score >= 10.0:
                keep_entity = True
                filter_reason = 'high_importance'
            elif frequency >= 15:
                keep_entity = True
                filter_reason = 'high_frequency'
            if keep_entity:
                filtered_entities.append({
                    'entity': entity,
                    'frequency': frequency,
                    'paper_count': paper_count,
                    'section_diversity': section_diversity,
                    'weighted_score': weighted_score,
                    'labels': list(entity_labels),
                    'filter_reason': filter_reason,
                    'importance_score': weighted_score + (frequency * 0.5) + (paper_count * 2)
                })
        filtered_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        
        reasons = Counter(e['filter_reason'] for e in filtered_entities)
        for reason, count in reasons.items():
            print(f"   {reason}: {count}")
        
        return filtered_entities

    
    
    
