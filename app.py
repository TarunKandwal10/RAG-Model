import streamlit as st
import os
import numpy as np
import logging
import sys
from dotenv import load_dotenv
from google import genai

# Configure extensive logging - DEBUG for app, WARNING for third-party libraries
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)

# Silence noisy third-party loggers
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('streamlit').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("CINFLEX AI APPLICATION STARTING")
logger.info("=" * 60)

# Load API key securely
logger.info("Loading environment variables from .env file")
load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if api_key:
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
    logger.info(f"API Key loaded successfully: {masked_key}")
else:
    logger.error("No API key found in environment variables!")

st.set_page_config(page_title="Cinflex AI", layout="centered")
logger.info("Streamlit page config set: page_title='Cinflex AI', layout='centered'")

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Hide the top-right Streamlit menu explicitly */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    
    /* Hide the 'Made with Streamlit' footer */
    footer {visibility: hidden !important;}
    
    /* Force the header background to be transparent but keep elements visible */
    header {background: transparent !important; visibility: visible !important;}
    
    /* Keep chat text sizing */
    .stChatMessage { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# Sleek sidebar
with st.sidebar:
    st.title("Cinflex AI")
    st.markdown("Your personal streaming assistant.")
    st.divider()
    st.markdown("**Status:** 🟢 Database Loaded Offline")
    st.markdown("- Zero API Cost on Launch")
logger.debug("Sidebar rendered")

if not api_key or "your" in api_key.lower():
    logger.error("Invalid or missing API key - showing error to user")
    st.error("Please put your real Google API Key in the .env file.")
else:
    logger.info("Initializing Gemini client")
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize Gemini client")
        st.error(f"Failed to initialize Gemini client: {e}")
        st.stop()

    # Load everything instantly from local storage with zero API calls
    @st.cache_resource
    def load_local_vectors():
        logger.info("Loading local vectors from disk (cached resource)")
        try:
            knowledge_path = "data/knowledge.txt"
            embeddings_path = "data/embeddings.npy"
            
            logger.debug(f"Reading knowledge file: {knowledge_path}")
            with open(knowledge_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.debug(f"Knowledge file size: {len(text)} characters")
            
            chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 10]
            logger.info(f"Loaded {len(chunks)} knowledge chunks")
            logger.debug(f"Sample chunk (first 100 chars): {chunks[0][:100] if chunks else 'N/A'}")
            
            logger.debug(f"Loading embeddings from: {embeddings_path}")
            embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings shape: {embeddings.shape}")
            logger.debug(f"Embeddings dtype: {embeddings.dtype}, first 5 values: {embeddings[0][:5] if len(embeddings) > 0 else 'N/A'}")
            
            return chunks, embeddings
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.exception("Error loading local vectors")
            raise

    try:
        logger.info("Loading cached vectors...")
        knowledge_chunks, chunk_embeddings = load_local_vectors()
        logger.info("Local vectors loaded successfully")
            
        user_query = st.chat_input("Ask a question about your Netflix dataset...")
        
        if user_query:
            logger.info(f"User query received: '{user_query[:100]}...' (length: {len(user_query)})")
            
            with st.chat_message("user"):
                st.write(user_query)
            logger.debug("User message displayed in chat")
            
            with st.chat_message("assistant"):
                try:
                    with st.status("Searching local database...", expanded=False) as status:
                        logger.info("Starting embedding generation for user query")
                        # Only 1 embedding call happens here per question!
                        query_response = client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=user_query,
                        )
                        query_embedding = np.array(query_response.embeddings[0].values)
                        logger.info(f"Query embedding generated: shape={query_embedding.shape}")
                        logger.debug(f"Query embedding sample: {query_embedding[:5]}")
                        
                        logger.info("Computing similarity scores against knowledge base")
                        scores = np.dot(chunk_embeddings, query_embedding)
                        logger.debug(f"Similarity scores computed: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
                        
                        top_indices = np.argsort(scores)[-10:][::-1] 
                        logger.info(f"Top {len(top_indices)} indices selected: {top_indices}")
                        logger.debug(f"Top scores: {[scores[i] for i in top_indices]}")
                        
                        retrieved_context = "\n\n".join([knowledge_chunks[i] for i in top_indices])
                        logger.info(f"Retrieved context length: {len(retrieved_context)} characters")
                        logger.debug(f"Retrieved context preview: {retrieved_context[:200]}...")
                        
                        status.update(label="Search complete!", state="complete")
                        logger.info("Search status updated to complete")
                    
                    # Call Gemini for the final text response
                    logger.info("Generating response with Gemini model")
                    prompt = f"Answer strictly based on this context:\n\n{retrieved_context}\n\nQuestion: {user_query}"
                    logger.debug(f"Prompt length: {len(prompt)} characters")
                    
                    chat_response = client.models.generate_content(
                        model="gemini-3.5-flash", 
                        contents=prompt
                    )
                    
                    logger.info("Gemini response received successfully")
                    logger.debug(f"Response length: {len(chat_response.text)} characters")
                    logger.debug(f"Response preview: {chat_response.text[:200]}...")
                    
                    st.write(chat_response.text)
                    logger.info("Response displayed to user")
                    
                except Exception as api_error:
                    error_msg = str(api_error)
                    logger.exception(f"API Error occurred: {error_msg}")
                    if "429" in error_msg:
                        # This will now show us exactly which limit (Daily or Per Minute) we hit!
                        logger.warning(f"Quota reached: {error_msg}")
                        st.error(f"⏳ **Quota Reached.** Google's exact error: {error_msg}")
                    elif "503" in error_msg:
                        logger.warning("Gemini servers busy (503)")
                        st.error("🤖 **Gemini Servers are Busy.** Please try resubmitting in a few seconds.")
                    else:
                        logger.error(f"API Error: {error_msg}")
                        st.error(f"API Error: {error_msg}")

    except Exception as e:
        logger.exception(f"System error loading local data: {e}")
        st.error(f"System error loading local data: {e}")

logger.info("Application rendering complete")