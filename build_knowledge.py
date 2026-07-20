import csv
import os
import numpy as np
import logging
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_knowledge.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("KNOWLEDGE BASE BUILDER STARTING")
logger.info("=" * 60)

# Load API key to generate embeddings
logger.info("Loading environment variables from .env file")
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if api_key:
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
    logger.info(f"API Key loaded successfully: {masked_key}")
else:
    logger.error("No API key found in environment variables!")

def build_knowledge_and_embeddings(input_csv, output_txt, output_npy):
    logger.info(f"Starting knowledge build process")
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Output text: {output_txt}")
    logger.info(f"Output embeddings: {output_npy}")
    
    if not api_key:
        logger.error("No API key found in your .env file.")
        print("❌ Error: No API key found in your .env file.")
        return

    logger.info("Initializing Gemini client for embedding generation")
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize Gemini client")
        print(f"❌ Error: Failed to initialize Gemini client: {e}")
        return

    logger.debug(f"Creating output directory for: {output_txt}")
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    logger.debug("Output directory ensured")
    
    chunks = []
    
    # 1. Read and format the CSV rows
    logger.info(f"Reading CSV file: {input_csv}")
    print("📖 Reading CSV file...")
    try:
        with open(input_csv, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            logger.debug(f"CSV headers: {reader.fieldnames}")
            row_count = 0
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
                row_count += 1
                
                if row_count % 1000 == 0:
                    logger.debug(f"Processed {row_count} rows...")
            
            logger.info(f"Total rows read from CSV: {row_count}")
            print(f"📊 Read {row_count} rows from CSV")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {input_csv}")
        print(f"❌ Error: CSV file not found: {input_csv}")
        return
    except Exception as e:
        logger.exception(f"Error reading CSV file: {e}")
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Slice to a safe number for the free tier offline build
    original_count = len(chunks)
    chunks = chunks[:80]
    logger.info(f"Sliced chunks from {original_count} to {len(chunks)} for free tier limit")
    print(f"📝 Using first {len(chunks)} chunks (limited for free tier)")
    
    # 2. Save text chunks to the knowledge file
    logger.info(f"Saving {len(chunks)} text chunks to {output_txt}")
    print(f"💾 Saving {len(chunks)} text chunks to {output_txt}...")
    try:
        with open(output_txt, mode='w', encoding='utf-8') as outfile:
            for i, chunk in enumerate(chunks):
                outfile.write(chunk + "\n\n")
                if i % 20 == 0:
                    logger.debug(f"Written chunk {i+1}/{len(chunks)}")
        logger.info(f"Successfully saved {len(chunks)} chunks to {output_txt}")
        print(f"✅ Text chunks saved successfully")
    except Exception as e:
        logger.exception(f"Error saving text chunks: {e}")
        print(f"❌ Error saving text chunks: {e}")
        return

    # 3. Generate embeddings offline via Google SDK
    logger.info("Starting embedding generation via Google API (this happens only ONCE)")
    print("🧠 Generating mathematical vectors from Google API (This happens only ONCE)...")
    try:
        logger.debug("Preparing content for embedding API")
        contents_for_embedding = [
            types.Content(parts=[types.Part.from_text(text=chunk)]) 
            for chunk in chunks
        ]
        logger.debug(f"Prepared {len(contents_for_embedding)} content items for embedding")
        
        logger.info("Calling Gemini embedding API with model: gemini-embedding-2")
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents_for_embedding,
        )
        logger.info("Embedding API response received")
        
        logger.debug(f"Number of embeddings returned: {len(response.embeddings)}")
        embeddings = np.array([e.values for e in response.embeddings])
        logger.info(f"Embeddings array created: shape={embeddings.shape}, dtype={embeddings.dtype}")
        logger.debug(f"First embedding sample (first 10 values): {embeddings[0][:10] if len(embeddings) > 0 else 'N/A'}")
        
    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        print(f"❌ Error generating embeddings: {e}")
        return
    
    # 4. Save vectors locally to a binary numpy file
    logger.info(f"Saving embeddings to {output_npy}")
    try:
        np.save(output_npy, embeddings)
        logger.info(f"Successfully saved embeddings to {output_npy}")
        print(f"✅ Successfully saved mathematical vectors to {output_npy}!")
    except Exception as e:
        logger.exception(f"Error saving embeddings: {e}")
        print(f"❌ Error saving embeddings: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("KNOWLEDGE BASE BUILD COMPLETED SUCCESSFULLY")
    logger.info(f"Chunks: {len(chunks)}, Embeddings shape: {embeddings.shape}")
    logger.info("=" * 60)
    print("🚀 Knowledge base is now fully compiled offline.")

if __name__ == "__main__":
    logger.info("Running build_knowledge_and_embeddings as main script")
    build_knowledge_and_embeddings("netflix_titles.csv", "data/knowledge.txt", "data/embeddings.npy")