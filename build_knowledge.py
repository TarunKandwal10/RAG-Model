import csv
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key to generate embeddings
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

def build_knowledge_and_embeddings(input_csv, output_txt, output_npy):
    if not api_key:
        print("❌ Error: No API key found in your .env file.")
        return

    client = genai.Client(api_key=api_key)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    chunks = []
    
    # 1. Read and format the CSV rows
    print("📖 Reading CSV file...")
    with open(input_csv, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            title = row.get('title') or 'Unknown'
            show_type = row.get('type') or 'Unknown'
            director = row.get('director') or 'Unknown'
            cast = row.get('cast') or 'Unknown'
            release_year = row.get('release_year') or 'Unknown'
            genre = row.get('listed_in') or 'Unknown'
            description = row.get('description') or 'No description'
            
            formatted_row = f"Type: {show_type}. Title: '{title}' ({release_year}). Genre: {genre}. Directed by: {director}. Cast: {cast}. Plot: {description}"
            chunks.append(formatted_row)
            
    # Slice to a safe number for the free tier offline build
    chunks = chunks[:80]
    
    # 2. Save text chunks to the knowledge file
    print(f"💾 Saving {len(chunks)} text chunks to {output_txt}...")
    with open(output_txt, mode='w', encoding='utf-8') as outfile:
        for chunk in chunks:
            outfile.write(chunk + "\n\n")

    # 3. Generate embeddings offline via Google SDK
    print("🧠 Generating mathematical vectors from Google API (This happens only ONCE)...")
    contents_for_embedding = [
        types.Content(parts=[types.Part.from_text(text=chunk)]) 
        for chunk in chunks
    ]
    
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents_for_embedding,
    )
    
    embeddings = np.array([e.values for e in response.embeddings])
    
    # 4. Save vectors locally to a binary numpy file
    np.save(output_npy, embeddings)
    print(f"✅ Successfully saved mathematical vectors to {output_npy}!")
    print("🚀 Knowledge base is now fully compiled offline.")

if __name__ == "__main__":
    build_knowledge_and_embeddings("netflix_titles.csv", "data/knowledge.txt", "data/embeddings.npy")