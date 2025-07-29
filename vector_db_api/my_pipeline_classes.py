# importing all needed libraries
import os
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sklearn.pipeline import Pipeline
from qdrant_client.models import PointStruct
import pickle



# Creating a class which performs cleaning on the pdf(s)

class PDFTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_numbers=False, remove_special=True, lower=True,
                 remove_graphics=True, remove_headers_footers=True):
        self.remove_numbers = remove_numbers
        self.remove_special = remove_special
        self.lower = lower
        self.remove_graphics = remove_graphics
        self.remove_headers_footers = remove_headers_footers

    def _extract_pages_text(self, pdf_path):
        """Extract text page by page to perform header/footer detection."""
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return pages

    def _detect_and_remove_headers_footers(self, pages_text):
        """Detect and remove repeated text snippets at top/bottom of pages (headers/footers)."""
        # Collect first and last lines of all pages
        first_lines = [page_text.split('\n')[0].strip() if page_text else '' for page_text in pages_text]
        last_lines = [page_text.split('\n')[-1].strip() if page_text else '' for page_text in pages_text]

        # Find most common header and footer lines
        header = self._most_common_nonempty(first_lines)
        footer = self._most_common_nonempty(last_lines)

        cleaned_pages = []
        for text in pages_text:
            lines = text.split('\n')
            # Remove header if it matches the detected header
            if lines and lines[0].strip() == header:
                lines = lines[1:]
            # Remove footer if it matches the detected footer
            if lines and lines[-1].strip() == footer:
                lines = lines[:-1]
            cleaned_pages.append('\n'.join(lines))
        return cleaned_pages

    def _most_common_nonempty(self, lines):
        """Return the most common non-empty line."""
        nonempty_lines = [line for line in lines if line]
        if not nonempty_lines:
            return ''
        counter = Counter(nonempty_lines)
        most_common_line, count = counter.most_common(1)[0]
        # Consider as header/footer only if repeated more than once
        return most_common_line if count > 1 else ''

    def _remove_graphics_text(self, text):
        """Attempt to remove charts, graphs, and tables based on keywords and patterns."""
        # Remove common figure/table references
        text = re.sub(r'\b(Figure|Fig|Table|Chart|Graph|Diagram|Plot|Image|Illustration)\s*\d+', '', text, flags=re.I)

        # Remove typical table structures with lines or many pipes, dashes etc.
        # A heuristic to find tabular data in text form
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            # If line looks like parts of tables or charts, remove
            if re.search(r'[\|\-\+═]+', line):  # lines with table borders
                continue
            # If line has many tab spaces or columns separated by multiple spaces
            if len(line.split()) > 3 and re.search(r'\s{3,}', line):
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)

    def _clean_text(self, text):
        if self.lower:
            text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace in the whole text
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        if self.remove_special:
            text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def fit(self, X, y=None):
      self.is_fitted_ = True
      return self

    def transform(self, folder_paths):
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]

        cleaned_texts = []

        for folder in folder_paths:
            for filename in os.listdir(folder):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(folder, filename)
                    try:
                        pages_text = self._extract_pages_text(pdf_path)
                        if self.remove_headers_footers:
                            pages_text = self._detect_and_remove_headers_footers(pages_text)
                        full_text = '\n'.join(pages_text)
                        if self.remove_graphics:
                            full_text = self._remove_graphics_text(full_text)
                        cleaned_text = self._clean_text(full_text)
                        cleaned_texts.append(cleaned_text)
                    except Exception as e:
                        print(f"Error processing {pdf_path}: {e}")
        return cleaned_texts




# Creating a class which converts text to embeddings

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Load model once at fit time
        self.model = SentenceTransformer(self.model_name)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # X is a list of cleaned strings (documents)
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(X, show_progress_bar=False)
        return np.array(embeddings)





# creating class which manages updating the vector store

class QdrantVectorStoreManager(BaseEstimator, TransformerMixin):
    def __init__(self, path, collection_name, vector_size):
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        self.status = ""

    def _connect(self):
        try:
            self.client = QdrantClient(path=self.path)
            return True
        except Exception as e:
            self.status = f"Error connecting to Qdrant: {e}"
            print(self.status)
            return False

    def _collection_exists(self):
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            self.status = f"Collection check error: {e}"
            print(self.status)
            return False

    def _create_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            return True
        except Exception as e:
            self.status = f"Create collection error: {e}"
            print(self.status)
            return False

    def fit(self, X, y=None):
      self.is_fitted_ = True
      return self

    def transform(self, X):
        try:
            if self.client is None and not self._connect():
                self.status = "Failed to connect to Qdrant store."
                print(self.status)
                return X

            if not self._collection_exists():
                if not self._create_collection():
                    self.status = "Failed to create collection."
                    print(self.status)
                    return X

            points = [
                PointStruct(id=int(idx), vector=vector.tolist())
                for idx, vector in enumerate(X)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            self.status = "Qdrant vector store updated successfully."
            print(self.status)

        except Exception as e:
            self.status = f"Failed to update Qdrant vector store: {e}"
            print(self.status)
        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        state['client'] = None  # Exclude unpickleable client connection
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # client will be None; will reconnect on next use
