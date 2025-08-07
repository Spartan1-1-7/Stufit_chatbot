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
- Vector database implementation for knowledge retrieval (Qdrant)
- Large Language Model (LLM) integration with PEFT fine-tuning
- Parameter Efficient Fine-Tuning (PEFT) techniques for medical domain adaptation
- Model merging and optimization techniques
- Interactive web interface development
- Comprehensive RAG (Retrieval-Augmented Generation) pipeline

## ✨ Features

### Core Functionality
- **Medical Report Analysis**: Analyze uploaded medical reports and lab results
- **Intelligent Chat Interface**: Interactive conversation with medical AI assistant
- **Knowledge Base Integration**: Access to comprehensive medical literature and guidelines
- **Real-time Processing**: Instant responses to medical queries
- **Document Upload**: Support for PDF medical documents

### Technical Features
- **Vector Database**: Qdrant integration for efficient similarity search and document retrieval
- **Large Language Model**: Llama-3-8B fine-tuned using PEFT for medical domain specialization
- **PEFT Integration**: Parameter Efficient Fine-Tuning with LoRA adapters for specialized medical responses
- **RAG Pipeline**: Complete Retrieval-Augmented Generation system combining vector search with LLM
- **Model Optimization**: 4-bit quantization with BitsAndBytesConfig for efficient inference
- **Document Processing**: Advanced PDF text extraction and cleaning
- **Responsive UI**: Modern, mobile-friendly Streamlit interface
- **API Gateway**: RESTful API for document ingestion and vector database management

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
3. **Vector Embedding**: Documents converted to embeddings using SentenceTransformers (all-MiniLM-L6-v2)
4. **Storage**: Embeddings stored in Qdrant vector database
5. **Query Processing**: User queries are embedded and matched against knowledge base using cosine similarity
6. **Retrieval**: Top-k relevant medical document chunks are retrieved
7. **Response Generation**: Fine-tuned Llama-3-8B generates contextual medical responses using RAG pipeline
8. **Model Optimization**: 4-bit quantization ensures efficient inference on consumer hardware

## 📁 Project Structure

```
Stufit_chatbot/
├── 📄 app.py                           # Main Streamlit application (legacy)
├── 📄 vector_db_interface.py           # Vector database upload interface
├── 📄 stufit_chatbot_environment.yml   # Conda environment configuration
├── 📄 test.yml                         # Test configuration
├── 📄 README.md                        # Project documentation
├── 📄 .env                             # Environment variables (HuggingFace tokens)
├── 📄 .gitignore                       # Git ignore file
│
├── 📂 rag_bot_files/                   # 🎯 Main RAG Bot Implementation
│   ├── 📄 final_chatbot.ipynb          # 🚀 Complete RAG chatbot with fine-tuned Llama-3
│   ├── 📄 LLM_model.py                 # Legacy LLM integration (Mixtral-based)
│   └── 📄 rag_bot_unmerged_model.ipynb # RAG bot with unmerged adapter
│
├── � fine_tunning/                    # Model Fine-tuning Components
│   ├── 📄 Stufit_LLM_Fine_tuning.ipynb        # Primary PEFT fine-tuning notebook
│   └── 📄 OL_approach_Stufit_LLM_Fine_tuning.ipynb # Alternative fine-tuning approach
│
├── 📂 merging_model_with_adapter/      # Model Optimization
│   └── 📄 merging-adapter-and-base-model.ipynb # LoRA adapter merging with base model
│
├── 📂 old_database/                    # Legacy Database Components
│   ├── 📄 finalvector.ipynb            # Original vector database creation
│   ├── 📄 ingest.ipynb                 # Document ingestion experiments
│   └── 📄 vectorized_medical_book_chunks.parquet # Processed medical data (legacy)
│
├── 📂 Books/                           # Medical Literature Collection
│   ├── adolescent-health.pdf
│   ├── bipolar-disorder-assessment-and-management-pdf-35109814379461_copy.pdf
│   ├── Blood Results in Clinical Practice_ A practical guide to interpreting blood test results - Graham Basten (2019, M&K Update Ltd).pdf
│   ├── constipation-in-children-and-young-people-diagnosis-and-management-pdf-975757753285_copy.pdf
│   ├── fatty acid intake.pdf
│   ├── guidelines-on-mental-health-promotive-and-preventive-interventions-for-adolescents-hat_copy.pdf
│   ├── hypertension.pdf
│   ├── IND301-20250415_copy.pdf
│   ├── NICD guideline mental health.pdf
│   ├── NICE 1.pdf
│   ├── obesity-clinical-assessment-and-management-pdf-75545363615173_copy.pdf
│   ├── obesity-in-adults-prevention-and-lifestyle-weight-management-programmes-pdf-75545293071301_copy.pdf
│   ├── overweight-and-obesity-management-pdf-66143959958725_copy.pdf
│   ├── physical-activity-exercise-referral-schemes-pdf-1996418406085_copy.pdf
│   ├── ROUTINE BLOOD RESULTS EXPLAINED - ANDREW BLANN (2022, CAMBRIDGE SCHOLARS PUB) (1).pdf
│   ├── WHO guideline for physics activity.pdf
│   ├── B1.pdf
│   └── new_books/                      # Extended medical reference collection
│       ├── Cecil Essentials of Medicine (Edward J. Wing MD  FACP  FIDSA (editor) etc.) (Z-Library).pdf
│       ├── CURRENT Medical Diagnosis and Treatment 2021 Maxine A. Papadakis, Stephen J. McPhee, Michael W. Rabow, ( etc.) (Z-Library).pdf
│       ├── Epidemiology (Leon Gordis) (Z-Library).pdf
│       └── ... (additional medical textbooks)
│
├── 📂 db_faiss/                        # Legacy FAISS Database (deprecated)
│   ├── faiss_index_chunk_text.faiss
│   ├── chunk_texts.pkl
│   └── chunk_lengths.pkl
│
├── 📂 vector_db_api/                   # FastAPI Backend Service
│   ├── 📄 app.py                       # FastAPI application
│   ├── 📄 my_pipeline_classes.py       # Custom ML pipeline components
│   ├── 📄 create_pipeline_pickle.py    # Pipeline serialization
│   ├── 📄 requirements.txt             # API dependencies
│   ├── 📄 test.py                      # API testing utilities
│   ├── 📄 qdrant_vector_db_pipeline.pkl # Serialized ML pipeline
│   ├── 📂 __pycache__/                 # Python cache files
│   ├── 📂 fit_data/                    # Training data for pipeline
│   ├── 📂 ingestion_source/            # Temporary upload directory
│   └── 📂 qdrant_db/                   # 🔥 Qdrant vector database (download required)
│
├── 📂 styles/                          # UI Styling and Themes
│   ├── 📄 __init__.py
│   ├── 📄 styling.py                   # Streamlit styling utilities
│   ├── 📄 styles.css                   # Custom CSS styles
│   └── 📂 __pycache__/                 # Python cache files
│
├── 📂 interface_assets/                # UI Components and Utilities
│   ├── 📄 responsive_styles.py         # Responsive design utilities
│   └── 📂 __pycache__/                 # Python cache files
│
├── 📂 media/                           # Static Assets
│   ├── stufit_logo.png
│   └── User_pfp.jpg
│
├── 📂 .streamlit/                      # Streamlit Configuration
│   └── config.toml                     # UI theme configuration
│
└── 📂 __pycache__/                     # Python Cache Files
    ├── LLM_model.cpython-312.pyc
    └── styling.cpython-312.pyc
```

```

## 🚀 Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/stufit-chatbot.git
cd stufit-chatbot

# Create and activate conda environment
conda env create -f stufit_chatbot_environment.yml
conda activate stufit_chatbot_env
```

#### Option B: Using pip
```bash
# Install dependencies
pip install streamlit torch transformers qdrant-client peft sentence-transformers
pip install bitsandbytes accelerate datasets
```

### 2. Download Vector Database
The Qdrant vector database must be downloaded separately due to size constraints:

1. **Download the database**: [Qdrant Database (Google Drive)](https://drive.google.com/file/d/1r09W1jXBdEfQ0V9bfXyXz2CJfVsB8M9S/view?usp=sharing)
2. **Extract** the downloaded file to `vector_db_api/qdrant_db/` directory
3. **Verify** the path: `vector_db_api/qdrant_db/` should contain the Qdrant database files

### 3. Configure HuggingFace Access
Create a `.env` file in the root directory with your HuggingFace token:
```bash
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 4. Run the Main Application

#### 🎯 Primary Method: Final RAG Chatbot (Recommended)
```bash
# Navigate to the main implementation
cd rag_bot_files
jupyter notebook final_chatbot.ipynb
```
Run all cells in the notebook to start the complete RAG system with fine-tuned Llama-3-8B.

#### Alternative: Streamlit Interface (Legacy)
```bash
streamlit run app.py
```

#### FastAPI Backend (Optional)
```bash
cd vector_db_api
python app.py
```

### 5. Start Chatting!
- For Jupyter notebook: Follow the cells to interact with the RAG system
- For Streamlit: Open browser to `http://localhost:8501`
- Upload medical reports or ask health-related questions
- Get AI-powered insights based on fine-tuned medical knowledge

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

# Download and setup Qdrant database (Required for Vector DB API)
# Download the database from: https://drive.google.com/file/d/1K8aX0lBSEQ6dMGFPpU3P2_X0A_0H7yYe/view?usp=sharing
# Extract the downloaded file and copy all contents to vector_db_api/qdrant_db/ folder

# Set up environment variables
echo "HUGGINGFACE_mixtrail_read_TOKEN=your_huggingface_token" > .env
```

### HuggingFace Token Setup
1. Create a free account at [HuggingFace](https://huggingface.co/)
2. Generate an access token in your account settings
3. Add the token to your `.env` file

### Qdrant Database Setup (Required for Vector DB API)
**Important**: To run the Vector Database API and pipeline, you must download the pre-built Qdrant database:

1. **Download the database**: 
   - Visit: https://drive.google.com/file/d/1K8aX0lBSEQ6dMGFPpU3P2_X0A_0H7yYe/view?usp=sharing
   - Download the compressed file

2. **Extract and setup**:
   ```bash
   # Extract the downloaded file
   # Copy all extracted contents to the qdrant_db folder
   cp -r /path/to/extracted/contents/* vector_db_api/qdrant_db/
   ```

3. **Verify setup**:
   - Ensure the `vector_db_api/qdrant_db/` folder contains the database files
   - The folder should not be empty after copying the contents

**Note**: Without this database setup, the Vector DB API will not function properly.

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

# Ensure Qdrant database is properly set up (see installation section)
# The qdrant_db folder should contain the downloaded database files

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 10000
```

**Prerequisites for API**:
- Qdrant database must be downloaded and placed in `vector_db_api/qdrant_db/`
- All dependencies from `requirements.txt` must be installed

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
- **LLM**: Meta Llama-3-8B with PEFT fine-tuning and 4-bit quantization
- **Fine-tuning**: Parameter Efficient Fine-Tuning (PEFT) with LoRA adapters for medical domain adaptation
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Databases**: Qdrant (primary), FAISS (legacy)
- **ML Framework**: PyTorch, Transformers, scikit-learn, PEFT, BitsAndBytesConfig

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

# Download and setup Qdrant database (Required)
# Download from: https://drive.google.com/file/d/1K8aX0lBSEQ6dMGFPpU3P2_X0A_0H7yYe/view?usp=sharing
# Extract and copy contents to vector_db_api/qdrant_db/

# Install additional development dependencies
pip install jupyter ipywidgets
```

### Fine-tuning the Model

```bash
# Run PEFT fine-tuning notebook
cd fine_tunning
jupyter notebook Stufit_LLM_Fine_tuning.ipynb

# Merge fine-tuned adapter with base model (optional)
cd ../merging_model_with_adapter
jupyter notebook merging-adapter-and-base-model.ipynb
```

### Training the Vector Database

```bash
# Process medical documents and create embeddings
python vector_db_api/create_pipeline_pickle.py

# Alternative: Use the legacy Jupyter notebook in old_database/
cd old_database
jupyter lab finalvector.ipynb
```

### Adding New Medical Documents

1. Place PDF files in the `Books/` directory
2. Run the vector database interface: `streamlit run vector_db_interface.py`
3. Upload documents through the web interface
4. The system will automatically process and add them to the knowledge base

### Customizing the Model

The main implementation is in `rag_bot_files/final_chatbot.ipynb`. You can customize:
- **LLM Configuration**: Change from Llama-3-8B to other models
- **PEFT Settings**: Modify LoRA adapter parameters and quantization settings
- **Fine-tuning Parameters**: Adjust training hyperparameters in `fine_tunning/` notebooks
- **RAG Pipeline**: Update retrieval parameters and prompt templates
- **Vector Database**: Switch between Qdrant and FAISS implementations
- **Embedding Models**: Change SentenceTransformers model for embeddings

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