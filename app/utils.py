import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("portfolio-chat")

# Initialize Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_DIMENSION = 768

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text using Gemini embedding model."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_rag_response(query: str) -> str:
    """
    RAG pipeline: embed query, retrieve context from Pinecone, generate response.
    """
    try:
        # Step 1: Embed the query
        query_embedding = get_embedding(query)
        if not query_embedding:
            return "I'm having a little trouble accessing my brain right now. Please try again!"

        # Step 2: Query Pinecone for top 5 matches
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        # Step 3: Extract context from matches
        context_chunks = []
        for match in results.matches:
            if match.metadata and "text" in match.metadata:
                context_chunks.append(match.metadata["text"])
        
        # Handle case where no context is found
        if not context_chunks:
            return "I couldn't find any specific details about that in Mrigank's portfolio, but feel free to ask about his patents, DASES, or other projects!"
        
        # Join chunks to create the context text
        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Step 4: Construct the system prompt
        system_prompt = f"""You are the Advanced AI Assistant for **Mrigank Singh**, a Full Stack AI Developer and Innovator. 
        Your goal is to impress recruiters and engineers by accurately showcasing Mrigank's technical depth, innovation, and leadership.

        ### CORE INSTRUCTIONS:
        1. **Identity:** You are NOT Mrigank. You are his digital assistant. Refer to him as "Mrigank" or "he".
        2. **Tone:** Professional, confident, and technically precise. Sound like a Software Engineer, not a marketing brochure.
        3. **Formatting:** Use **Markdown** to make answers readable. 
           - Use **bold** for key technologies or metrics.
           - Use `bullet points` for lists (skills, projects).
           - Do not output large walls of text; break it up.
        4. **Source of Truth:** Answer ONLY based on the "CONTEXT" provided below. Do not make up facts. 
           - If the answer isn't in the context, say: "I don't have that specific detail, but I can tell you about his patents, his projects or more about him."

        ### CRITICAL BEHAVIORS:
        - **Recruiters:** If asked about hiring, availability, or contact info, explicitly provide his **Email** and **LinkedIn** from the context.
        - **Patents:** If asked about innovation, ALWAYS mention his 3 filed patents (Terms & Conditions AI, LexiBot, MealMatch).
        - **Group Projects:** Credit **Konal Puri and Aviral Khanna** for DASES/UPES Career Platform. Specify Mrigank's role (Mobile App/Frontend).
        - **Technical Depth:** Mention specific algorithms (e.g., "Knapsack Pruning", "Isolation Forests", "Regex Chunking") to show engineering depth.

        ### CONTEXT FROM KNOWLEDGE BASE:
        {context_text}
        """
        
        # Step 5: Generate response using Gemini
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                types.Content(
                    role="user", 
                    parts=[
                        types.Part.from_text(text=system_prompt + "\n\nUser Question: " + query)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        return response.text

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return "I'm encountering a temporary issue connecting to the knowledge base. Please try again in a moment."