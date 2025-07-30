from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, shutil, pickle, sys
import fitz

# === Pipeline/dependency paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "qdrant_vector_db_pipeline.pkl")
SOURCE_DIR = os.path.join(BASE_DIR, "ingestion_source")
FIT_DIR = os.path.join(BASE_DIR, "fit_data")

sys.path.append(BASE_DIR)

from my_pipeline_classes import PDFTextCleaner, EmbeddingTransformer, QdrantVectorStoreManager

# === Load and fit the pipeline ===
with open(PIPELINE_PATH, "rb") as f:
    pipeline = pickle.load(f)
pipeline.fit(FIT_DIR)

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    filename = os.path.join(SOURCE_DIR, file.filename)
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        pipeline.transform(SOURCE_DIR)
        vec_store_step = pipeline.named_steps['qdrant_store']
        status_msg = getattr(vec_store_step, 'status', 'No status available.')
        os.remove(filename)
        return JSONResponse(content={
            "status": "success" if "success" in status_msg.lower() else "failure",
            "message": status_msg
        })
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        })

@app.get("/")
async def root():
    return {"status": "running"}
