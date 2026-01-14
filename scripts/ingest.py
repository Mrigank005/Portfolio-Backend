import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("portfolio-chat")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
DATA_FILE = Path(__file__).parent.parent / "data" / "knowledge_base.txt"


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "output_dimensionality": EMBEDDING_DIMENSION
        }
    )
    return response.embeddings[0].values


def load_and_chunk(file_path: Path) -> list[str]:
    """Load text file and split into chunks by double newlines."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newlines
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks


def main():
    print("=" * 50)
    print("Portfolio Knowledge Base Ingestion Script")
    print("=" * 50)
    
    # Step 1: Load and chunk the data
    print(f"\n[1/3] Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        sys.exit(1)
    
    chunks = load_and_chunk(DATA_FILE)
    print(f"      Loaded {len(chunks)} chunks")
    
    # Step 2: Generate embeddings and prepare vectors
    print(f"\n[2/3] Generating embeddings...")
    vectors = []
    
    for i, chunk in enumerate(chunks):
        print(f"      Processing chunk {i + 1}/{len(chunks)}...", end="\r")
        
        embedding = get_embedding(chunk)
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    print(f"      Generated {len(vectors)} embeddings" + " " * 20)
    
    # Step 3: Upsert to Pinecone
    print(f"\n[3/3] Upserting to Pinecone...")
    
    # Upsert in batches of 100 (Pinecone best practice)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"      Upserted batch {i // batch_size + 1}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Knowledge base ingested!")
    print(f"Total vectors: {len(vectors)}")
    print("=" * 50)


if __name__ == "__main__":
    main()