import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import pickle  # or use your method for storing FAISS index
import numpy as np
from sentence_transformers import SentenceTransformer


print(torch.__version__)
# Load the .env file
load_dotenv()

# Fetch the token
token = os.getenv("HUGGINGFACE_mixtral_TOKEN")

# Load FAISS index and metadata
faiss_index = faiss.read_index("db_faiss/faiss_index_chunk_text.faiss")  # your FAISS file
with open("db_faiss/chunk_texts.pkl", "rb") as f:
    documents = pickle.load(f)  # your list of text chunks corresponding to vectors
# print("FAISS index dimension:", faiss_index.d)


# Load the embedding model (same one used during indexing)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # or the one you used


# Load Mixtral model & tokenizer
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Create text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Prompt builder
def build_prompt(user_query, retrieved_chunks):
    joined_chunks = "\n\n".join(retrieved_chunks)
    instruction = f"""
You are a helpful medical assistant. Analyze the following medical report and retrieved expert medical knowledge. Identify any possible health issues or important insights.

Medical Report:
{user_query}

Relevant Medical Knowledge:
{joined_chunks}

Answer (Only use the above relevant knowledge. If not found, reply "I don't know."):
"""
    return f"<s>[INST] {instruction.strip()} [/INST]"


# Vector search + generation
def generate_response(user_query, top_k=3):
    query_embedding = embedding_model.encode([user_query])
    # print("Query embedding shape:", np.array(query_embedding).shape)
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    
    # print("Indices:", indices)
    # print("Type:", type(indices))
    # print("Shape:", np.shape(indices))

    # retrieved_chunks = [documents[i] for i in indices[0] if i != -1]
    retrieved_chunks = [documents[i] for i in indices[0] if i != -1 and i < len(documents)]

    if not retrieved_chunks:
        return "I don't know."

    prompt = build_prompt(user_query, retrieved_chunks)
    response = text_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return response[0]['generated_text']


if __name__ == "__main__":
    query = "What is a heart-attack?"
    answer = generate_response(query)
    print("Generated Answer:\n", answer)
