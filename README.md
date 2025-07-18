# Data Indexing Service

A robust, production-ready data indexing pipeline that processes documents from MongoDB, applies advanced NLP enrichment techniques, generates vector embeddings, and stores them in a Qdrant vector database for semantic search and retrieval.

## 🚀 Features

- **Document Processing**: Load documents from MongoDB collections
- **Advanced Text Chunking**: Split documents into optimal chunks using tiktoken encoding
- **NLP Enrichment**: 
  - Text cleaning and normalization
  - Unicode normalization
  - Spelling correction
  - Stopword removal
  - Keyword extraction
- **Vector Embeddings**: Generate high-quality embeddings using FlagEmbedding models
- **Vector Storage**: Store enriched chunks with metadata in Qdrant vector database
- **Scalable Pipeline**: Modular architecture for easy extension and maintenance

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- MongoDB instance
- Qdrant vector database
- CUDA-compatible GPU (recommended for embeddings)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## ⚙️ Configuration

Create a `.env` file in the project root with the following environment variables:

```bash
# MongoDB Configuration
MONGO_URI=mongodb://localhost
MONGO_PORT=27017
MONGO_DB_NAME=your_database_name
MONGO_COLLECTION_NAME=your_collection_name

# Text Processing Configuration
TOKENIZATION_CHUNK_SIZE=1000
TOKENIZATION_CHUNK_OVERLAP=200
TOKENIZATION_ENCODING_NAME=cl100k_base

# Embedding Configuration
EMBEDDER_MODEL_NAME=BAAI/bge-large-en-v1.5

# Vector Database Configuration
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_COLLECTION_NAME=document_chunks
VECTOR_DB_EMBEDDING_SIZE=1024

# Alternative Qdrant Configuration (used in retriever)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=document_chunks

# Retrieval Configuration
RETRIEVER_TOP_K=5

# NLP Configuration
KEYWORD_EXTRACTION_TOP_N=10

# Prompt Configuration
INCLUDE_METADATA=True
MAX_CONTEXT_CHARS=4000
PROMPT_TYPE=qa
```

## 🚀 Usage

### Command Line Interface

Run the complete indexing pipeline:

```bash
python cli.py
```

### Programmatic Usage

```python
from data_indexing.pipeline import run_indexing_job

# Run the complete pipeline
run_indexing_job()
```

### Individual Components

```python
from data_indexing.mongo_loader import get_document_content
from data_indexing.chunker import chunk_text
from data_indexing.chunk_enricher import enrich_chunks
from data_indexing.embedder import embed_chunks
from data_indexing.storage import upsert_chunks

# Load documents
documents = get_document_content()

# Process documents step by step
documents = chunk_text(documents)
chunk_records = enrich_chunks(documents)
chunk_records = embed_chunks(chunk_records)
upsert_chunks(chunk_records)
```

## 🏗 Architecture

The service follows a modular pipeline architecture:

```
MongoDB → Chunker → Enricher → Embedder → Qdrant
```

### Core Components

- **`mongo_loader.py`**: MongoDB document retrieval
- **`chunker.py`**: Text splitting using LangChain with tiktoken
- **`chunk_enricher.py`**: NLP preprocessing and enrichment
- **`embedder.py`**: Vector embedding generation using FlagEmbedding
- **`storage.py`**: Qdrant vector database operations
- **`pipeline.py`**: Pipeline orchestration
- **`utils.py`**: Configuration and utility functions

### Data Flow

1. **Load**: Documents are retrieved from MongoDB
2. **Chunk**: Documents are split into manageable chunks
3. **Enrich**: Chunks undergo NLP processing:
   - Text cleaning and normalization
   - Spelling correction
   - Stopword removal
   - Keyword extraction
4. **Embed**: Vector embeddings are generated for each chunk
5. **Store**: Enriched chunks with embeddings are stored in Qdrant

### Document Schema

Input documents should have the following structure:

```json
{
  "id": "unique_document_id",
  "content": "document text content",
  "source": "document_source",
  "type": "document_type",
  "title": "document_title",
  "location": "document_location",
  "created_at": "2025-01-01T00:00:00Z",
  "fetched_at": "2025-01-01T00:00:00Z",
  "language": "en"
}
```

## 📝 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MONGO_URI` | MongoDB connection URI | - | ✅ |
| `MONGO_PORT` | MongoDB port | - | ✅ |
| `MONGO_DB_NAME` | MongoDB database name | - | ✅ |
| `MONGO_COLLECTION_NAME` | MongoDB collection name | - | ✅ |
| `TOKENIZATION_CHUNK_SIZE` | Maximum tokens per chunk | 1000 | ✅ |
| `TOKENIZATION_CHUNK_OVERLAP` | Token overlap between chunks | 200 | ✅ |
| `TOKENIZATION_ENCODING_NAME` | Tiktoken encoding | cl100k_base | ✅ |
| `EMBEDDER_MODEL_NAME` | HuggingFace model for embeddings | - | ✅ |
| `VECTOR_DB_URL` | Qdrant database URL | - | ✅ |
| `VECTOR_DB_COLLECTION_NAME` | Qdrant collection name | - | ✅ |
| `VECTOR_DB_EMBEDDING_SIZE` | Embedding vector dimension | - | ✅ |
| `KEYWORD_EXTRACTION_TOP_N` | Number of keywords to extract | 10 | ✅ |
| `RETRIEVER_TOP_K` | Number of top relevant chunks to retrieve | - | ✅ |
| `QDRANT_URL` | Alternative Qdrant URL (used in retriever) | - | ✅ |
| `QDRANT_COLLECTION_NAME` | Alternative collection name (used in retriever) | - | ✅ |
| `INCLUDE_METADATA` | Whether to include metadata in prompts | - | ✅ |
| `MAX_CONTEXT_CHARS` | Maximum characters in context for prompts | - | ✅ |
| `PROMPT_TYPE` | Type of prompt template to use | - | ✅ |

## 🔧 Development

### Project Structure

```
data-indexing-service/
├── cli.py                 # Command-line interface
├── data_indexing/         # Main package
│   ├── __init__.py
│   ├── chunk_enricher.py  # NLP enrichment
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Vector embeddings
│   ├── mongo_loader.py    # MongoDB operations
│   ├── pipeline.py        # Pipeline orchestration
│   ├── storage.py         # Vector database operations
│   └── utils.py           # Utilities
├── tests/                 # Test directory
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=data_indexing
```

### Code Quality

```bash
# Format code
black data_indexing/

# Lint code
flake8 data_indexing/

# Type checking
mypy data_indexing/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance Considerations

- **Memory Usage**: Large embedding models require significant RAM
- **GPU Acceleration**: CUDA-compatible GPU recommended for embedding generation
- **Batch Processing**: Consider batch processing for large document collections
- **Vector Database**: Ensure Qdrant has sufficient resources for your data volume

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection**: Verify MongoDB is running and accessible
2. **Qdrant Connection**: Ensure Qdrant service is running
3. **Memory Issues**: Reduce batch size or upgrade system resources
4. **NLTK Data**: Download required NLTK datasets
5. **GPU Issues**: Ensure CUDA drivers are properly installed

### Logging

The service uses Python's logging module. Adjust log levels in `cli.py`:

```python
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for verbose output
```

## 📈 Monitoring

Monitor the following metrics:

- Processing speed (documents/minute)
- Memory usage during embedding generation
- Vector database storage utilization
- Error rates and types

## 🔒 Security

- Store sensitive credentials in environment variables
- Use connection encryption for production databases
- Implement proper access controls for MongoDB and Qdrant
- Regular security updates for dependencies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for text processing utilities
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for high-quality embeddings
- [Qdrant](https://qdrant.tech/) for vector database capabilities
- [MongoDB](https://www.mongodb.com/) for document storage

---

**Version**: 0.1.0  
**Author**: Ishank Vasania  
**Maintained**: ✅ Active 