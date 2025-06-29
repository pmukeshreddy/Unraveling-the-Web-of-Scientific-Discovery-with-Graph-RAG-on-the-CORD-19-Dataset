import os
import json
import re
import pandas as pd
from tqdm.notebook import tqdm
import textwrap


def parse_json_data(file_path):
    try:
        with open(file_path,"r") as f:
            data = json.load(f)

        paper_id = data.get("paper_id","")
        paper_title = data.get("metadata",{}).get("title","no title that's it")

        document_parts = []

        abstract_part = "\n".join([item.get("text","") for item in data.get("abstract","")])

        if abstract_part:
            document_parts.append({"paper_id":paper_id,"paper_title":paper_title,"section_title":"Abstract","section_text":abstract_part})

        sections = {}

        current_section_title = ""

        for item in data.get("body_text",[]):
            if item.get("section") and item["section"].strip():
                current_section_title = item["section"].strip()
            if current_section_title not in sections:
                sections[current_section_title] = []
            sections[current_section_title].append(item.get("text",""))

        for sec_title , text_title in sections.items():
            full_section_text = "\n\n".join(text_title)
            if full_section_text:
                document_parts.append({
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'section_title': sec_title,
                    'section_text': full_section_text
                })
        return document_parts
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Could not process file {os.path.basename(file_path)}: {e}")
        return []

all_sections_data = []
# Get a list of all json files. Use a slice for faster testing.
json_files = os.listdir(PDF_JSON_PATH)

for filename in tqdm(json_files, desc="Parsing Papers into Sections (Corrected)"):
    file_path = os.path.join(PDF_JSON_PATH, filename)
    # Call the new, robust function
    sections = parse_json_data(file_path)
    if sections:
        all_sections_data.extend(sections)

# Convert the list of dictionaries into a pandas DataFrame
df_sections = pd.DataFrame(all_sections_data)

print(f"Successfully parsed {len(df_sections)} sections from {df_sections['paper_id'].nunique()} papers.")
df_sections.info()
print("\nVerifying section titles:")
print(df_sections['section_title'].value_counts().head(10))

def clean_text(text):
    text = re.sub(r'\[\d+(?:, ?\d+)*(?:-\d+)*\]',"",text) # for [1]
    text = re.sub(r'([^\s\w.,!?-]|\s_)',"",text) #for ©
    text = re.sub(r'\s+',"",text).strip()
    return text

def chunk_text(text,chunk_size=300,overlap=40):
    if not isinstance(text,str) or not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step_size = chunk_size - overlap

    for i in range(0,len(words),step_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break
    return chunks

print("Creating granular text chunks for detailed Knowledge Graph extraction...")

chunked_data = []
# Iterate through each section in the df_sections DataFrame
for index, row in tqdm(df_sections.iterrows(), total=df_sections.shape[0], desc="Chunking Sections"):
    paper_id = row['paper_id']
    paper_title = row['paper_title']
    section_title = row['section_title']
    
    # First, clean the full text of the section
    cleaned_text = clean_text(row['section_text'])
    
    # Second, break the cleaned text into smaller, overlapping chunks
    text_chunks = chunk_text(cleaned_text, chunk_size=300, overlap=40)
    
    # Third, create a dictionary for each chunk and add it to our list
    for i, chunk in enumerate(text_chunks):
        # Sanitize the section title to create a safe, filesystem-friendly string for the ID
        safe_section_title = re.sub(r'\W+', '_', section_title).lower()
        chunk_id = f"{paper_id}_{safe_section_title}_{i}"
        
        chunked_data.append({
            'paper_id': paper_id,
            'paper_title': paper_title,
            'section_title': section_title,
            'chunk_id': chunk_id,
            'chunk_text': chunk
        })

# Convert the list of dictionaries into our final DataFrame for this output
df_chunks = pd.DataFrame(chunked_data)

print(f"\n[Output 1 Complete] Created `df_chunks` DataFrame with {len(df_chunks):,} small text chunks.")
df_chunks.info()

# Display the first few rows to verify the output
print("\nSample of `df_chunks`: \n")
display(df_chunks.head())




