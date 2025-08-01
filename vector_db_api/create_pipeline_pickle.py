import sys
import os
import pickle
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my_pipeline_classes import PDFTextCleaner, TextChunker, EmbeddingTransformer, QdrantVectorStoreManager

fit_dir = "fit_data"

pipeline = Pipeline([
    ('pdf_cleaner', PDFTextCleaner()),
    ('text_chunker', TextChunker(max_chunk_chars=2000, chunk_overlap=800)),  # tuned for medical text
    ('embedder', EmbeddingTransformer()),  # device always set at transform time!
    ('qdrant_store', QdrantVectorStoreManager(
        path="qdrant_db",
        collection_name="your_collection",
        vector_size=384  # (384 for all-MiniLM-L6-v2; update to your embedding size if changed)
    )),
])

pipeline.fit(fit_dir)

with open('qdrant_vector_db_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Pipeline pickle created successfully.")
