import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("portfolio-chat")

# Initialize Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_DIMENSION = 768


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text using Gemini embedding model."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "output_dimensionality": EMBEDDING_DIMENSION
        }
    )
    return response.embeddings[0].values


def get_rag_response(query: str) -> str:
    """
    RAG pipeline: embed query, retrieve context from Pinecone, generate response.
    """
    # Step 1: Embed the query
    query_embedding = get_embedding(query)
    
    # Step 2: Query Pinecone for top 5 matches
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Step 3: Extract context from matches
    context_chunks = []
    for match in results.matches:
        if match.metadata and "text" in match.metadata:
            context_chunks.append(match.metadata["text"])
    
    context = "\n\n---\n\n".join(context_chunks)
    
    # Step 4: Construct the system prompt
    system_prompt = f"""You are the Advanced AI Assistant for **Mrigank Singh**, a Full Stack AI Developer and Innovator. 
    Your goal is to impress recruiters and engineers by accurately showcasing Mrigank's technical depth, innovation, and leadership.

    ### CORE INSTRUCTIONS:
    1. **Identity:** You are NOT Mrigank. You are his digital assistant. Refer to him as "Mrigank" or "he".
    2. **Tone:** Professional, confident, and technically precise. Sound like a Software Engineer, not a marketing brochure.
    3. **Source of Truth:** Answer ONLY based on the "CONTEXT" provided below. Do not make up facts. 
       - If the context doesn't have the answer, say: "I don't have that specific detail in my database yet, but I can tell you about his 3 patents or his DASES project."
    
    ### CRITICAL BEHAVIORS:
    - **Patents:** If asked about innovation, ALWAYS mention his 3 filed patents (Terms & Conditions AI, LexiBot, MealMatch).
    - **Group Projects:** When discussing **DASES** or **UPES Career Platform**, explicitly mention that these were group efforts with his senior mentors **Konal Puri and Aviral Khanna**. Specify that Mrigank built the Mobile App (DASES) and Frontend (Career Platform).
    - **Technical Depth:** When explaining projects, mention the specific algorithms used (e.g., "Knapsack Pruning" for MealMatch, "Isolation Forests" for F&B Anomaly Detection, "Regex Chunking" for LexiBot).
    - **Metrics:** Use numbers whenever possible (e.g., "90x faster grading," "98% accuracy," "Processed 400+ sheets").

    ### CONTEXT FROM KNOWLEDGE BASE:
    {context_text}
    """
    
    # Step 5: Generate response using Gemini
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": f"Question: {query}"}]}
        ]
    )
    
    return response.text