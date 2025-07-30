# Stufit Report Analyzer 🏥🤖

> **Internship Project Repository**  
> An AI-powered medical report analysis chatbot developed during an internship program

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Technologies Used](#-technologies-used)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

Stufit Report Analyzer is an intelligent chatbot designed to analyze medical reports and provide insights based on expert medical knowledge. The system uses advanced natural language processing and machine learning techniques to understand medical queries and respond with relevant, evidence-based information.

**This is an internship project that demonstrates:**
- Medical document processing and analysis
- Vector database implementation for knowledge retrieval
- Large Language Model (LLM) integration with PEFT fine-tuning
- Parameter Efficient Fine-Tuning (PEFT) techniques for domain adaptation
- Interactive web interface development
- RESTful API design

## ✨ Features

### Core Functionality
- **Medical Report Analysis**: Analyze uploaded medical reports and lab results
- **Intelligent Chat Interface**: Interactive conversation with medical AI assistant
- **Knowledge Base Integration**: Access to comprehensive medical literature and guidelines
- **Real-time Processing**: Instant responses to medical queries
- **Document Upload**: Support for PDF medical documents

### Technical Features
- **Vector Database**: FAISS and Qdrant integration for efficient similarity search
- **Large Language Model**: Mixtral-8x7B fine-tuned using PEFT for medical domain
- **PEFT Integration**: Parameter Efficient Fine-Tuning for specialized medical responses
- **Document Processing**: Advanced PDF text extraction and cleaning
- **Responsive UI**: Modern, mobile-friendly interface
- **API Gateway**: RESTful API for document ingestion

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Vector DB     │
│   Frontend      │◄──►│   Backend       │◄──►│   (FAISS/       │
│                 │    │                 │    │    Qdrant)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   Document      │    │   Medical       │
│   Processing    │    │   Processing    │    │   Knowledge     │
│                 │    │   Pipeline      │    │   Base          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Document Ingestion**: Medical PDFs are uploaded via the web interface
2. **Text Processing**: Advanced cleaning and chunking of medical documents
3. **Vector Embedding**: Documents converted to embeddings using SentenceTransformers
4. **Storage**: Embeddings stored in vector databases (FAISS/Qdrant)
5. **Query Processing**: User queries are embedded and matched against knowledge base
6. **Response Generation**: Mixtral LLM generates contextual medical responses

## 📁 Project Structure

```
Stufit_chatbot/
├── 📄 app.py                           # Main Streamlit application
├── 📄 LLM_model.py                     # LLM integration and response generation
├── 📄 vector_db_interface.py           # Vector database upload interface
├── 📄 stufit_chatbot_environment.yml   # Conda environment configuration
├── 📄 vectorized_medical_book_chunks.parquet  # Processed medical data
│
├── 📂 Books/                           # Medical literature and guidelines
│   ├── adolescent-health.pdf
│   ├── bipolar-disorder-assessment-and-management.pdf
│   ├── Blood Results in Clinical Practice.pdf
│   ├── hypertension.pdf
│   ├── obesity-clinical-assessment-and-management.pdf
│   ├── WHO guideline for physics activity.pdf
│   └── new_books/                      # Extended medical reference collection
│       ├── Cecil Essentials of Medicine.pdf
│       ├── CURRENT Medical Diagnosis and Treatment 2021.pdf
│       └── Harrisons Principles of Internal Medicine.pdf
│
├── 📂 db_faiss/                        # FAISS vector database files
│   ├── faiss_index_chunk_text.faiss
│   ├── chunk_texts.pkl
│   └── chunk_lengths.pkl
│
├── 📂 vector_db_api/                   # FastAPI backend service
│   ├── 📄 app.py                       # FastAPI application
│   ├── 📄 my_pipeline_classes.py       # Custom ML pipeline components
│   ├── 📄 create_pipeline_pickle.py    # Pipeline serialization
│   ├── 📄 requirements.txt             # API dependencies
│   ├── 📄 qdrant_vector_db_pipeline.pkl # Serialized ML pipeline
│   ├── 📂 fit_data/                    # Training data for pipeline
│   ├── 📂 ingestion_source/            # Temporary upload directory
│   └── 📂 qdrant_db/                   # Qdrant vector database
│
├── 📂 styles/                          # UI styling and themes
│   ├── 📄 styling.py                   # Streamlit styling utilities
│   └── 📄 styles.css                   # Custom CSS styles
│
├── 📂 interface_assets/                # UI components and utilities
│   └── 📄 responsive_styles.py         # Responsive design utilities
│
├── 📂 media/                           # Static assets
│   ├── stufit_logo.png
│   └── User_pfp.jpg
│
├── 📄 finalvector.ipynb                # Data processing and analysis notebook
├── 📄 ingest.ipynb                     # Document ingestion experiments
└── 📄 test.yml                         # Test configuration
```

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Conda (recommended) or pip
- CUDA-compatible GPU (optional, for faster processing)

### Method 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/Stufit_chatbot.git
cd Stufit_chatbot

# Create and activate conda environment
conda env create -f stufit_chatbot_environment.yml
conda activate stufit_chatbot

# Set up environment variables
echo "HUGGINGFACE_mixtrail_read_TOKEN=your_huggingface_token" > .env
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/Stufit_chatbot.git
cd Stufit_chatbot

# Create virtual environment
python -m venv stufit_env
source stufit_env/bin/activate  # On Windows: stufit_env\Scripts\activate

# Install dependencies
pip install streamlit torch transformers sentence-transformers
pip install faiss-cpu python-dotenv huggingface_hub
pip install langchain-community accelerate peft

# Install API dependencies
cd vector_db_api
pip install -r requirements.txt
cd ..

# Set up environment variables
echo "HUGGINGFACE_mixtrail_read_TOKEN=your_huggingface_token" > .env
```

### HuggingFace Token Setup
1. Create a free account at [HuggingFace](https://huggingface.co/)
2. Generate an access token in your account settings
3. Add the token to your `.env` file

## 💻 Usage

### Running the Main Application

```bash
# Start the Streamlit application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Running the Vector Database API

```bash
# Navigate to API directory
cd vector_db_api

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 10000
```

The API will be available at `http://localhost:10000`

### Using the Vector Database Interface

```bash
# Run the document upload interface
streamlit run vector_db_interface.py
```

## 🔧 API Documentation

### FastAPI Endpoints

#### Upload PDF Document
```http
POST /upload-pdf/
Content-Type: multipart/form-data

Parameters:
- file: PDF file to process and add to vector database

Response:
{
    "status": "success|failure|error",
    "message": "Processing status message"
}
```

#### Health Check
```http
GET /

Response:
{
    "status": "running"
}
```

### Usage Examples

```python
import requests

# Upload a medical document
with open('medical_report.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:10000/upload-pdf/',
        files={'file': f}
    )
    print(response.json())
```

## 🛠️ Technologies Used

### Machine Learning & AI
- **LLM**: Mixtral-8x7B-Instruct-v0.1 (Mistral AI) fine-tuned with PEFT
- **Fine-tuning**: Parameter Efficient Fine-Tuning (PEFT) for medical domain adaptation
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Databases**: FAISS, Qdrant
- **ML Framework**: PyTorch, Transformers, scikit-learn, PEFT

### Backend & APIs
- **Web Framework**: FastAPI
- **Document Processing**: PyPDF2, PyMuPDF (fitz)
- **Data Processing**: pandas, numpy
- **Pipeline**: scikit-learn Pipeline

### Frontend & UI
- **Web Interface**: Streamlit
- **Styling**: Custom CSS, responsive design
- **File Handling**: Multi-file upload support

### Data & Storage
- **Vector Storage**: FAISS index files, Qdrant database
- **Serialization**: Pickle, Parquet
- **Configuration**: YAML, TOML

### Development Tools
- **Environment**: Conda, pip
- **Notebooks**: Jupyter Lab/Notebook
- **Version Control**: Git

## 🔬 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/Stufit_chatbot.git
cd Stufit_chatbot

# Install in development mode
conda env create -f stufit_chatbot_environment.yml
conda activate stufit_chatbot

# Install additional development dependencies
pip install jupyter ipywidgets
```

### Training the Vector Database

```bash
# Process medical documents and create embeddings
python vector_db_api/create_pipeline_pickle.py

# Alternative: Use the Jupyter notebook
jupyter lab finalvector.ipynb
```

### Adding New Medical Documents

1. Place PDF files in the `Books/` directory
2. Run the vector database interface: `streamlit run vector_db_interface.py`
3. Upload documents through the web interface
4. The system will automatically process and add them to the knowledge base

### Customizing the Model

Edit `LLM_model.py` to:
- Change the LLM model
- Modify PEFT configuration and adapters
- Adjust fine-tuning parameters
- Update prompt templates
- Adjust retrieval parameters
- Update embedding models

## 🤝 Contributing

This is an internship project, but contributions and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Test new features thoroughly
- Update documentation as needed

## 📄 License

This project is created as part of an internship program. Please contact the repository owner for licensing information.

## 📞 Contact & Support

For questions, issues, or collaboration opportunities related to this internship project:

- Create an issue in this repository
- Contact the development team
- Check the documentation in the `docs/` folder (if available)

---

**Note**: This is an educational/internship project designed to demonstrate AI application in healthcare. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified medical professionals for health-related concerns.

## 🙏 Acknowledgments

- Medical literature and guidelines providers
- Open-source ML/AI community
- Internship program supervisors and mentors
- HuggingFace for model hosting
- Streamlit and FastAPI communities

---

*Developed during internship program - Showcasing AI applications in healthcare* 🏥✨