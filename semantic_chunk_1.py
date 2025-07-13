def parse_json_data(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        paper_id = data.get("paper_id", "")
        paper_title = data.get("metadata", {}).get("title", "no title that's it")
        if not paper_title or not paper_title.strip():
            paper_title = "no title that's it"
        
        document_parts = []
        valid_titles_lower = [
            'abstract', 'introduction', 'background', 'objective', 'methods', 
            'materials and methods', 'methodology', 'results', 'findings', 
            'discussion', 'conclusion', 'conclusions', 'summary', 'references'
        ]

        abstract_part = "\\n".join([item.get("text", "") for item in data.get("abstract", [])])
        if abstract_part:
            document_parts.append({"paper_id": paper_id, "paper_title": paper_title, "section_title": "Abstract", "section_text": abstract_part})

        sections = {}
        current_section_title = "Introduction"
        for item in data.get("body_text", []):
            potential_title = item.get("section", "").strip()
            if potential_title.lower() in valid_titles_lower:
                current_section_title = potential_title.title()
            if current_section_title not in sections:
                sections[current_section_title] = []
            sections[current_section_title].append(item.get("text", ""))

        for sec_title, text_list in sections.items():
            full_section_text = "\\n\\n".join(text_list)
            if full_section_text:
                document_parts.append({
                    'paper_id': paper_id, 'paper_title': paper_title,
                    'section_title': sec_title, 'section_text': full_section_text
                })
        return document_parts
    except Exception as e:
        # print(f"Could not process file {os.path.basename(file_path)}: {e}")
        return []

import re

def clean_text(text):
    """
    Cleans text by removing citation markers, special characters, and normalizing whitespace.
    """
    # 1. Removes citation numbers like [1], [2, 3], [4-7]
    text = re.sub(r'\\[\\d+(?:, ?\\d+)*(?:-\\d+)*\\]', '', text)
    
    # 2. Removes characters that are not standard letters, numbers, or basic punctuation
    text = re.sub(r'([^\\s\\w.,!?-]|\\s_)', '', text)
    
    # 3. Correctly condenses any whitespace (spaces, newlines) into a single space
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def chunk_text(text, chunk_size=300, overlap=40):
    if not isinstance(text, str) or not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks
