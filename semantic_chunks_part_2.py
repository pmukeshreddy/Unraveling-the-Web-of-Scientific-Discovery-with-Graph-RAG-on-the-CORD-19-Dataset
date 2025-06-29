import requests
import time

from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GEMMINI_API_KEY = secrets.get_secret("**************")

api_key = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=****************"
HEADERS =  {
    "Content-Type": "application/json"
}

CHUNK_CHAR_LIMIT = 8000

def call_gemini_api(prompt):
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(api_key,headers=HEADERS, json=payload)
        response.raise_for_status()
        
        # Parse the JSON response and extract the generated text
        # This path is specific to the Gemini API response structure
        result = response.json()
        
        # Add a safety check in case the response is blocked or empty
        if 'candidates' not in result or not result['candidates']:
             return "Error: Response was blocked or empty."
             
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error: Network request failed: {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Could not parse API response: {e}. Response: {result}"


def summarize_section_map_reduce_requests(text,section_title):
    if len(text) < 8000:
        if len(text.split()) < 20:
            return "Section too short to summarize."
        direct_prompt = f"""You are a biomedical research assistant. Provide a concise, one-paragraph summary of the following text from a scientific paper.

        Section Title: "{section_title}"
        Text to Summarize: "{text}"
        Concise Summary:"""
        return call_gemini_api(direct_prompt)
    else:
        text_chunks = [text[i:i+8000] for i in range(0,len(text),8000)]
        map_summaries = []
        for i in enumerate(text_chunks):
            map_prompt = f"The following is chunk {i+1}/{len(text_chunks)} of a long scientific section titled '{section_title}'. Summarize this specific chunk's main points. Text: \"{chunk}\" Concise Summary of chunk:"
            summary = call_gemini_api(map_prompt)
            if not summary.startswith("Error:"):
                map_summaries.append(summary)
            time.sleep(1)
        if not map_summaries:
            return "Error: Could not generate any summaries in the map step."
        print("  - Reducing: Combining summaries into a final version...")
        combined_summaries = "\n\n".join(map_summaries)
        reduce_prompt = f"""You are a biomedical research assistant. The following are several summaries from different parts of a single scientific section titled "{section_title}". Synthesize these into one final, coherent paragraph. Partial Summaries: \"{combined_summaries}\" Final, Synthesized Summary:"""
        return call_gemini_api(reduce_prompt) 
