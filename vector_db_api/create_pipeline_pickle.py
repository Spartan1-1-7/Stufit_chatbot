import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_pipeline_classes import PDFTextCleaner,TextChunker, EmbeddingTransformer, QdrantVectorStoreManager
import pickle
from sklearn.pipeline import Pipeline
import fitz

# Paths used for fitting
fit_dir = "fit_data"

# Construct your pipeline exactly as you do in your app
pipeline = Pipeline([
    ('pdf_cleaner', PDFTextCleaner()),
    ('text_chunker', TextChunker(max_chunk_chars=1000, chunk_overlap=200)),
    ('embedder', EmbeddingTransformer()),
    ('qdrant_store', QdrantVectorStoreManager(path="qdrant_db", collection_name="your_collection", vector_size=384)),
])

pipeline.fit(fit_dir)

with open('qdrant_vector_db_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Pipeline pickle created successfully.")