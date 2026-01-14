# Portfolio Chatbot API

A RAG (Retrieval-Augmented Generation) based chatbot API for Mrigank Singh's portfolio, powered by FastAPI, Google Gemini, and Pinecone.

## 🚀 Features

- **FastAPI Backend**: High-performance async API with automatic OpenAPI documentation
- **RAG Architecture**: Combines semantic search with LLM generation for accurate, context-aware responses
- **Vector Search**: Uses Pinecone for efficient similarity search over portfolio knowledge base
- **Google Gemini Integration**: Leverages Gemini 2.0 Flash for responses and embedding generation
- **CORS Enabled**: Ready for frontend integration from any origin
- **Production Ready**: Deployed on Render with health check endpoints

## 📋 Prerequisites

- Python 3.9+
- Pinecone account and API key
- Google AI Studio API key
- (Optional) Render account for deployment

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Portfolio-Backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## 📊 Project Structure

```
Portfolio-Backend/
├── app/
│   ├── __init__.py          # Application initialization
│   ├── main.py              # FastAPI app and endpoints
│   └── utils.py             # RAG pipeline and helper functions
├── data/
│   └── knowledge_base.txt   # Portfolio information source
├── scripts/
│   └── ingest.py            # Data ingestion script for Pinecone
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Pinecone Setup

1. Create a Pinecone index named `portfolio-chat`
2. Configure the index with:
   - Dimension: 768 (matches Gemini embedding model)
   - Metric: Cosine similarity

### Knowledge Base

The knowledge base is stored in [data/knowledge_base.txt](data/knowledge_base.txt). The file should contain information about the portfolio owner, structured in chunks separated by double newlines.

## 📤 Data Ingestion

Before running the API, you need to ingest the knowledge base into Pinecone:

```bash
python scripts/ingest.py
```

This script will:
1. Load and chunk the knowledge base text file
2. Generate embeddings using Google Gemini
3. Upsert vectors to Pinecone in batches

## 🚀 Running the Application

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

## 📡 API Endpoints

### Health Check
```http
GET /
```
Returns the API status.

**Response:**
```json
{
  "status": "ok"
}
```

### Chat
```http
POST /chat
```

Send a message to the chatbot and receive a context-aware response.

**Request Body:**
```json
{
  "message": "Tell me about Mrigank's projects"
}
```

**Response:**
```json
{
  "response": "Mrigank has worked on several impressive projects including..."
}
```

**Error Responses:**
- `400 Bad Request`: Empty message
- `500 Internal Server Error`: Error generating response

## 🧠 How RAG Works

1. **Query Embedding**: User's question is converted to a 768-dimensional vector using Gemini
2. **Semantic Search**: Top 5 most relevant chunks are retrieved from Pinecone
3. **Context Assembly**: Retrieved chunks are combined into a context string
4. **LLM Generation**: Context and query are sent to Gemini 2.0 Flash for response generation
5. **Response**: Contextually accurate answer is returned to the user

## 🌐 Deployment

This project is configured for deployment on Render. The [render.yaml](render.yaml) file contains the deployment configuration.

### Deploy to Render

1. Connect your GitHub repository to Render
2. Render will automatically detect the `render.yaml` configuration
3. Add environment variables in Render dashboard:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
4. Deploy!

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio API key | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Yes |

## 📦 Dependencies

- **FastAPI** (0.115.0): Modern web framework for building APIs
- **Uvicorn** (0.30.6): ASGI server for running FastAPI
- **Google GenAI** (1.0.0): Google's generative AI client library
- **Pinecone Client** (5.0.1): Vector database client
- **Pydantic** (2.9.2): Data validation using Python type annotations
- **Python-dotenv** (1.0.1): Environment variable management

## 📝 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🛡️ Security Considerations

- Store API keys securely in environment variables, never commit them to version control
- In production, restrict CORS origins to trusted domains
- Consider rate limiting for the `/chat` endpoint
- Implement authentication for sensitive deployments

## 🐛 Troubleshooting

### Common Issues

**Issue**: `PINECONE_API_KEY not found`
- **Solution**: Ensure `.env` file exists and contains the required API keys

**Issue**: `Index 'portfolio-chat' not found`
- **Solution**: Create the Pinecone index before running the ingestion script

**Issue**: `No context chunks found`
- **Solution**: Run the ingestion script to populate the Pinecone index

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of a personal portfolio. All rights reserved.

## 📧 Contact

For questions or collaboration opportunities, reach out to Mrigank Singh through his portfolio website.

---

**Built with** ❤️ **using FastAPI, Google Gemini, and Pinecone**
