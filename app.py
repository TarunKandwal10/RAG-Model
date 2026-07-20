import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load API key securely
load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

st.set_page_config(page_title="Pure Python RAG", layout="centered")

# Custom CSS for UI styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

if not api_key or "your" in api_key.lower():
    st.error("Please put your real Google API Key in the .env file.")
else:
    client = genai.Client(api_key=api_key)

    # Load everything instantly from local storage with zero API calls
    @st.cache_resource
    def load_local_vectors():
        with open("data/knowledge.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 10]
        embeddings = np.load("data/embeddings.npy")
        return chunks, embeddings

    try:
        knowledge_chunks, chunk_embeddings = load_local_vectors()
            
        user_query = st.chat_input("Ask a question about your Netflix dataset...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
                
            with st.chat_message("assistant"):
                try:
                    with st.status("Searching local database...", expanded=False) as status:
                        # Only 1 embedding call happens here per question!
                        query_response = client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=user_query,
                        )
                        query_embedding = np.array(query_response.embeddings[0].values)
                        
                        scores = np.dot(chunk_embeddings, query_embedding)
                        top_indices = np.argsort(scores)[-10:][::-1] 
                        
                        retrieved_context = "\n\n".join([knowledge_chunks[i] for i in top_indices])
                        status.update(label="Search complete!", state="complete")
                    
                    # Call Gemini for the final text response
                    prompt = f"Answer strictly based on this context:\n\n{retrieved_context}\n\nQuestion: {user_query}"
                    
                    chat_response = client.models.generate_content(
                        model="gemini-3.5-flash", 
                        contents=prompt
                    )
                    
                    st.write(chat_response.text)
                    
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "429" in error_msg:
                        # This will now show us exactly which limit (Daily or Per Minute) we hit!
                        st.error(f"⏳ **Quota Reached.** Google's exact error: {error_msg}")
                    elif "503" in error_msg:
                        st.error("🤖 **Gemini Servers are Busy.** Please try resubmitting in a few seconds.")
                    else:
                        st.error(f"API Error: {error_msg}")

    except Exception as e:
        st.error(f"System error loading local data: {e}")