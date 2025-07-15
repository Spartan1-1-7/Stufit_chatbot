from huggingface_hub import InferenceClient

# 🔑 Add your Hugging Face token here
token = "your_huggingface_token_here"

# Choose model
client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=token
)

def build_prompt(user_query, retrieved_chunks):
    joined_chunks = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful medical assistant. Analyze the following medical report and retrieved expert medical knowledge. Identify any possible health issues or important insights.

Medical Report:
{user_query}

Relevant Medical Knowledge:
{joined_chunks}

Answer:"""
    return prompt

def generate_response(user_query, retrieved_chunks):
    prompt = build_prompt(user_query, retrieved_chunks)
    response = client.text_generation(prompt=prompt, max_new_tokens=300)
    return response
