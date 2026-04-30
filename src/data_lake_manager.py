"""
Task 2 — Data Storage Architecture [10 Marks]

MinIO-based data lake with three storage layers:
- Raw: Original files with metadata
- Processed: Cleaned Parquet files, vocabulary, TF-IDF matrix
- Embeddings: Versioned Word2Vec files

Provides DataLakeManager class with local filesystem fallback.

TECHNICAL JUSTIFICATION (400 words):
We selected MinIO as our storage backend for PakSentinel's data lake architecture after 
evaluating all five options: AWS S3 + Athena, Google Cloud Storage, MinIO, PostgreSQL + pgvector, 
and MongoDB Atlas. Our decision was driven by three core criteria: scalability, cost, and 
query capability.

SCALABILITY: MinIO is a high-performance, S3-compatible object storage system designed for 
large-scale data infrastructure. It supports horizontal scaling across multiple nodes and 
handles objects ranging from kilobytes (metadata files) to gigabytes (embedding models) 
uniformly. For PakSentinel's three-layer architecture (Raw, Processed, Embeddings), MinIO's 
bucket-based organization maps naturally to our storage tiers. Unlike PostgreSQL + pgvector, 
which requires schema management and struggles with binary blob storage at scale, MinIO treats 
all objects as first-class citizens. Compared to AWS S3 and GCS, MinIO can be deployed 
on-premises or in Docker, eliminating vendor lock-in and enabling deployment in environments 
with data sovereignty requirements — critical for a Pakistani misinformation detection system.

COST: MinIO is open-source (Apache 2.0 license) and runs entirely on local infrastructure 
or institutional servers, making it the most cost-effective option. AWS S3 and GCS incur 
per-request charges, data transfer costs, and storage fees that scale with dataset size — 
problematic for an academic project that may process millions of TF-IDF computations and 
embedding vectors during experimentation. MongoDB Atlas's free tier (512MB) is insufficient 
for our 44,000+ article corpus with embeddings. MinIO eliminates all cloud costs while 
providing identical S3 API compatibility, allowing seamless migration to AWS S3 in production 
if needed.

QUERY CAPABILITY: MinIO supports S3 Select for server-side filtering of Parquet and CSV files, 
enabling efficient data retrieval without downloading entire objects. Combined with our Parquet 
storage format for processed data, we achieve columnar query performance comparable to Athena 
but without the per-query cost. For embedding retrieval, MinIO's versioning system (with 
version IDs on each object) enables reproducible experiments — we can track exactly which 
Word2Vec model version produced specific results. The Python minio client provides a clean 
API for programmatic access, and our DataLakeManager class abstracts storage operations 
behind a consistent interface.

Our three-layer architecture ensures clear data lineage: Raw preserves original data integrity, 
Processed contains reproducible cleaned outputs, and Embeddings stores versioned model 
artifacts. Each layer uses appropriate formats — JSON metadata sidecars for Raw, Parquet for 
Processed, and binary .model files for Embeddings — optimized for their respective access 
patterns.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent / "data"
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

BUCKETS = {
    'raw': 'paksentinel-raw',
    'processed': 'paksentinel-processed',
    'embeddings': 'paksentinel-embeddings',
}


class DataLakeManager:
    """
    Data Lake Manager with three storage layers.
    
    Supports MinIO (S3-compatible) when available, falls back to local filesystem.
    
    Storage Layers:
        - Raw: Original files with JSON metadata sidecars
        - Processed: Cleaned Parquet files, vocabulary pickles, TF-IDF matrices
        - Embeddings: Versioned Word2Vec model files
    """
    
    def __init__(self, use_minio: bool = True):
        """
        Initialize DataLakeManager.
        
        Args:
            use_minio: If True and minio package is available, use MinIO backend.
                      Otherwise, use local filesystem.
        """
        self.use_minio = use_minio and MINIO_AVAILABLE
        self.client = None
        
        if self.use_minio:
            try:
                self.client = Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=MINIO_SECURE,
                )
                # Create buckets if they don't exist
                for bucket_name in BUCKETS.values():
                    if not self.client.bucket_exists(bucket_name):
                        self.client.make_bucket(bucket_name)
                print("[DataLakeManager] Connected to MinIO backend")
            except Exception as e:
                print(f"[DataLakeManager] MinIO connection failed: {e}")
                print("[DataLakeManager] Falling back to local filesystem")
                self.use_minio = False
                self.client = None
        
        if not self.use_minio:
            # Setup local filesystem directories
            for layer in ['raw', 'processed', 'embeddings']:
                (BASE_DIR / layer).mkdir(parents=True, exist_ok=True)
            print("[DataLakeManager] Using local filesystem backend")
    
    # ──────────────────────────────────────────────────
    #  Raw Layer
    # ──────────────────────────────────────────────────
    def upload_raw(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """
        Upload a raw data file with associated metadata.
        
        Args:
            file_path: Path to the file to upload
            metadata: Dictionary with metadata (source, date, description, etc.)
            
        Returns:
            Storage path/key of the uploaded file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Add standard metadata fields
        metadata.update({
            'upload_timestamp': datetime.now().isoformat(),
            'original_filename': file_path.name,
            'file_size_bytes': file_path.stat().st_size,
            'md5_hash': self._compute_md5(file_path),
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"{timestamp}/{file_path.name}"
        metadata_name = f"{timestamp}/{file_path.stem}_metadata.json"
        
        if self.use_minio:
            # Upload file to MinIO
            self.client.fput_object(
                BUCKETS['raw'], object_name, str(file_path),
            )
            # Upload metadata sidecar
            meta_bytes = json.dumps(metadata, indent=2).encode('utf-8')
            from io import BytesIO
            self.client.put_object(
                BUCKETS['raw'], metadata_name,
                BytesIO(meta_bytes), len(meta_bytes),
                content_type='application/json',
            )
        else:
            # Local filesystem
            dest_dir = BASE_DIR / "raw" / timestamp
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(file_path, dest_dir / file_path.name)
            
            # Write metadata
            with open(dest_dir / f"{file_path.stem}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"  [RAW] Uploaded: {object_name}")
        return object_name
    
    # ──────────────────────────────────────────────────
    #  Processed Layer
    # ──────────────────────────────────────────────────
    def upload_processed(self, data: Any, name: str, version: str, 
                         data_type: str = 'dataframe') -> str:
        """
        Upload processed data to the processed storage layer.
        
        Args:
            data: The data to store (DataFrame, dict, scipy sparse matrix, etc.)
            name: Name identifier for the data (e.g., 'cleaned_dataset', 'tfidf_matrix')
            version: Version string (e.g., 'v1.0', 'v2.1')
            data_type: Type of data ('dataframe', 'vocabulary', 'tfidf_matrix', 'pickle')
            
        Returns:
            Storage path/key
        """
        object_name = f"{version}/{name}"
        
        if data_type == 'dataframe':
            object_name += '.parquet'
            if self.use_minio:
                temp_path = BASE_DIR / "processed" / f"_temp_{name}.parquet"
                data.to_parquet(temp_path, index=False)
                self.client.fput_object(BUCKETS['processed'], object_name, str(temp_path))
                temp_path.unlink()
            else:
                dest = BASE_DIR / "processed" / version
                dest.mkdir(parents=True, exist_ok=True)
                data.to_parquet(dest / f"{name}.parquet", index=False)
                
        elif data_type in ('vocabulary', 'tfidf_matrix', 'pickle'):
            object_name += '.pkl'
            if self.use_minio:
                temp_path = BASE_DIR / "processed" / f"_temp_{name}.pkl"
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                self.client.fput_object(BUCKETS['processed'], object_name, str(temp_path))
                temp_path.unlink()
            else:
                dest = BASE_DIR / "processed" / version
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / f"{name}.pkl", 'wb') as f:
                    pickle.dump(data, f)
        
        print(f"  [PROCESSED] Uploaded: {object_name}")
        return object_name
    
    # ──────────────────────────────────────────────────
    #  Embeddings Layer
    # ──────────────────────────────────────────────────
    def upload_embeddings(self, model_path: str, version: str, 
                          model_name: str = 'word2vec') -> str:
        """
        Upload a Word2Vec model to the embeddings layer.
        
        Args:
            model_path: Path to the .model file
            version: Version string
            model_name: Name of the model ('word2vec_cbow', 'word2vec_skipgram')
            
        Returns:
            Storage path/key
        """
        model_path = Path(model_path)
        object_name = f"{version}/{model_name}.model"
        
        if self.use_minio:
            self.client.fput_object(BUCKETS['embeddings'], object_name, str(model_path))
        else:
            dest = BASE_DIR / "embeddings" / version
            dest.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(model_path, dest / f"{model_name}.model")
            # Also copy associated files (.model.wv.vectors_ngrams.npy etc.)
            for ext_file in model_path.parent.glob(f"{model_path.name}*"):
                if ext_file != model_path:
                    shutil.copy2(ext_file, dest / ext_file.name)
        
        print(f"  [EMBEDDINGS] Uploaded: {object_name}")
        return object_name
    
    # ──────────────────────────────────────────────────
    #  Fetch & Query
    # ──────────────────────────────────────────────────
    def fetch_for_training(self, version: str = 'latest', 
                           name: str = 'cleaned_dataset') -> pd.DataFrame:
        """
        Fetch processed dataset for model training.
        
        Args:
            version: Version to fetch ('latest' or specific version string)
            name: Name of the dataset to fetch
            
        Returns:
            DataFrame ready for training
        """
        if version == 'latest':
            versions = self.list_versions(layer='processed')
            if not versions:
                raise FileNotFoundError("No processed data versions found")
            version = sorted(versions)[-1]
        
        if self.use_minio:
            object_name = f"{version}/{name}.parquet"
            temp_path = BASE_DIR / "processed" / f"_fetch_{name}.parquet"
            self.client.fget_object(BUCKETS['processed'], object_name, str(temp_path))
            df = pd.read_parquet(temp_path)
            temp_path.unlink()
        else:
            parquet_path = BASE_DIR / "processed" / version / f"{name}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Dataset not found: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        
        print(f"  [FETCH] Retrieved: {version}/{name} ({len(df)} rows)")
        return df
    
    def list_versions(self, layer: str = 'processed') -> List[str]:
        """
        List all available data versions in a storage layer.
        
        Args:
            layer: Storage layer ('raw', 'processed', 'embeddings')
            
        Returns:
            List of version strings
        """
        versions = set()
        
        if self.use_minio:
            bucket = BUCKETS[layer]
            objects = self.client.list_objects(bucket, recursive=True)
            for obj in objects:
                parts = obj.object_name.split('/')
                if len(parts) > 1:
                    versions.add(parts[0])
        else:
            layer_dir = BASE_DIR / layer
            if layer_dir.exists():
                for item in layer_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('_'):
                        versions.add(item.name)
        
        versions = sorted(versions)
        print(f"  [VERSIONS] {layer}: {versions}")
        return versions
    
    # ──────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────
    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()


# ──────────────────────────────────────────────────────────
#  Main / Demo
# ──────────────────────────────────────────────────────────
def run_task2():
    """Demonstrate DataLakeManager functionality."""
    print("=" * 60)
    print("TASK 2: DATA STORAGE ARCHITECTURE")
    print("=" * 60)
    
    # Initialize with local fallback (MinIO requires docker-compose up)
    dlm = DataLakeManager(use_minio=False)
    
    # List current versions
    dlm.list_versions('raw')
    dlm.list_versions('processed')
    dlm.list_versions('embeddings')
    
    return dlm


if __name__ == "__main__":
    dlm = run_task2()
    print("\n✓ Task 2 complete. DataLakeManager initialized.")
