#!/usr/bin/env python3
"""
Searchable Image Database using SigLIP 2 and SQLite-vec

This script provides two modes:
1. Scan mode: Process images from a directory and store embeddings
2. Search mode: Search for similar images using text or image queries

NOTE: This code was generated with AI assistance.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import hashlib
import json
import re
from collections import defaultdict

# ============================================================================
# Configuration Loading - Reads from config.json in project root
# ============================================================================
def load_config() -> Dict[str, str]:
    """
    Load configuration from config.json.
    
    Lookup order:
    1) Next to this script (./config.json)
    2) One directory up (../config.json) so you can keep a private config outside the publishable folder.
    """
    script_dir = Path(__file__).parent.absolute()
    candidates = [
        script_dir / "config.json",
        script_dir.parent / "config.json",
    ]
    
    for config_path in candidates:
        if not config_path.exists():
            continue
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config.json at {config_path}: {e}")
            print("Using default configuration.")
    
    # Default configuration if config.json doesn't exist
    return {
        "database_dir": "",
        "model_cache_dir": "models",
        "results_dir": "results",
        "thumbnails_dir": "thumbnails"
    }

def resolve_path(config_path: str, base_dir: Path) -> str:
    """Resolve a path from config - use as-is if absolute, otherwise join with base directory."""
    if not config_path:
        return ""
    path = Path(config_path)
    # If path is already absolute, use it as-is
    if path.is_absolute():
        return str(path)
    # Otherwise, join with base directory (parent of code folder for outputs)
    return str(base_dir / path)

def resolve_db_dir(config_dir: str, base_dir: Path) -> str:
    """Resolve a database directory from config; if empty, fall back to parent of database_path or base_dir."""
    if config_dir:
        return resolve_path(config_dir, base_dir)
    # Back-compat: infer from database_path if present
    db_path = _CONFIG.get("database_path", "")
    if db_path:
        resolved = resolve_path(db_path, base_dir)
        try:
            return str(Path(resolved).parent)
        except Exception:
            pass
    return str(base_dir)

def list_db_files(db_dir: str) -> List[str]:
    """List .db files in db_dir (non-recursive)."""
    try:
        p = Path(db_dir)
        if not p.exists() or not p.is_dir():
            return []
        return sorted([f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".db"])
    except Exception:
        return []

def resolve_db_path(args_db: Optional[str], args_db_name: Optional[str], db_dir: str) -> str:
    """
    Resolve a DB path.
    - If args_db is provided, use it (absolute or relative).
    - Else if args_db_name is provided, resolve under db_dir.
    - Else raise ValueError.
    """
    if args_db:
        return str(Path(args_db))
    if args_db_name:
        name = args_db_name
        if not name.lower().endswith(".db"):
            name += ".db"
        return str(Path(db_dir) / name)
    raise ValueError("No database specified")

# Load configuration
_CONFIG = load_config()
# For outputs (results, etc.), use parent directory. For absolute paths in config, they'll be used as-is.
_OUTPUT_BASE = Path(__file__).parent.absolute().parent

# Configuration variables (paths relative to output base, or absolute if specified)
DEFAULT_DB_DIR = resolve_db_dir(_CONFIG.get("database_dir", ""), _OUTPUT_BASE)
DEFAULT_DB_PATH = str(Path(DEFAULT_DB_DIR) / "image_database.db")
DEFAULT_MODEL_CACHE_DIR = resolve_path(_CONFIG.get("model_cache_dir", "models"), _OUTPUT_BASE)
DEFAULT_RESULTS_DIR = resolve_path(_CONFIG.get("results_dir", "results"), _OUTPUT_BASE)
DEFAULT_THUMBNAILS_DIR = resolve_path(_CONFIG.get("thumbnails_dir", "thumbnails"), _OUTPUT_BASE)
# ============================================================================

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel
from tqdm import tqdm
import numpy as np
import sqlite_vec

# PDF support - use PyMuPDF (fitz) which doesn't require poppler
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not available. PDF support disabled. Install with: pip install PyMuPDF")

print("Python libraries loaded successfully.", flush=True)

# Increase PIL image size limit to handle very large images (default is ~89MP, we'll allow up to 500MP)
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 megapixels


class ImageDatabase:
    """Manages the image database with SigLIP 2 embeddings and SQLite-vec."""
    
    def __init__(self, db_path: str = None, model_cache_dir: str = None):
        print("="*60, flush=True)
        print("Initializing Image Database", flush=True)
        print("="*60, flush=True)
        
        # Default paths
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        if model_cache_dir is None:
            model_cache_dir = DEFAULT_MODEL_CACHE_DIR
        
        self.db_path = db_path
        self.model_cache_dir = model_cache_dir
        
        print(f"Database path: {self.db_path}")
        print(f"Model cache directory: {self.model_cache_dir}")
        
        # Create directories if they don't exist
        print("Creating directories...")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            print(f"  [OK] Created model cache directory: {model_cache_dir}")
        
        # Check device
        print("\nChecking compute device...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        if self.device.type == "cuda":
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  [OK] Using CPU (CUDA not available)")
        
        print(f"  [OK] Data type: {self.dtype}")
        
        # Initialize model
        print(f"\nLoading SigLIP 2 model...")
        print(f"  Model: google/siglip2-so400m-patch14-224")
        print(f"  Device: {self.device}")
        print(f"  Precision: {self.dtype}")
        
        # Try to load from local directory first, then fall back to HuggingFace
        local_model_path = os.path.join(model_cache_dir, "google--siglip2-so400m-patch14-224") if model_cache_dir else None
        model_name = "google/siglip2-so400m-patch14-224"
        
        if local_model_path and os.path.exists(local_model_path):
            print(f"  Loading from local cache: {local_model_path}")
            # Try explicit Siglip classes first to avoid Auto class confusion
            try:
                print("    Loading processor...")
                self.processor = SiglipProcessor.from_pretrained(local_model_path)
                print("    [OK] Processor loaded")
                print("    Loading model...")
                self.model = SiglipModel.from_pretrained(local_model_path).to(self.device).to(self.dtype)
                print("    [OK] Model loaded")
            except Exception as e:
                print(f"    Error with explicit classes, trying Auto: {e}")
                print("    Loading processor (Auto)...")
                self.processor = AutoProcessor.from_pretrained(local_model_path)
                print("    Loading model (Auto)...")
                self.model = AutoModel.from_pretrained(local_model_path).to(self.device).to(self.dtype)
        else:
            # Use default cache if model_cache_dir is None, otherwise use custom
            cache_kwargs = {}
            if model_cache_dir:
                cache_kwargs['cache_dir'] = model_cache_dir
                print(f"  Model cache directory: {model_cache_dir}")
            
            # Try to load from HuggingFace
            try:
                print("    Downloading/loading processor from HuggingFace...")
                self.processor = AutoProcessor.from_pretrained(model_name, **cache_kwargs)
                print("    [OK] Processor loaded")
                print("    Downloading/loading model from HuggingFace (this may take a while)...")
                self.model = AutoModel.from_pretrained(model_name, **cache_kwargs).to(self.device).to(self.dtype)
                print("    [OK] Model loaded")
            except Exception as e:
                print(f"    [X] Error loading model: {e}")
                raise
        
        print("  Setting model to evaluation mode...")
        self.model.eval()
        print("  [OK] Model ready")
        
        # Expected embedding dimension for SO400M
        self.embedding_dim = 1152
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        # Initialize database
        print("\nInitializing database...")
        self._init_database()
        print("="*60)
        print("Initialization complete!")
        print("="*60 + "\n")
    
    def _init_database(self):
        """Initialize SQLite database with sqlite-vec extension."""
        print(f"  Connecting to database: {self.db_path}")
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        print("  [OK] Database connection established")
        
        # Enable WAL mode for better concurrent access (reads can happen during writes)
        print("  Enabling WAL mode for concurrent access...")
        conn.execute("PRAGMA journal_mode=WAL")
        print("  [OK] WAL mode enabled")
        
        # Enable extension loading (required for sqlite-vec)
        print("  Enabling extension loading...")
        conn.enable_load_extension(True)
        print("  [OK] Extension loading enabled")
        
        # Load sqlite-vec extension
        print("  Loading sqlite-vec extension...")
        try:
            sqlite_vec.load(conn)
            print("  [OK] sqlite-vec extension loaded")
        except Exception as e:
            print(f"  [X] ERROR: Could not load sqlite-vec extension: {e}")
            print("  Please ensure sqlite-vec is installed: pip install sqlite-vec")
            sys.exit(1)
        
        cursor = conn.cursor()
        
        # Create metadata table
        print("  Creating images metadata table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                last_modified REAL NOT NULL,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  [OK] Images table ready")
        
        # Create vec0 virtual table for embeddings
        # Only create if it doesn't exist (don't drop existing data!)
        print(f"  Creating vec0 virtual table (embedding dimension: {self.embedding_dim})...")
        try:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec0 USING vec0(
                    embedding float[{self.embedding_dim}]
                )
            """)
            print("  [OK] vec0 virtual table ready")
        except sqlite3.OperationalError as e:
            if "no such module: vec0" in str(e).lower() or "no such module" in str(e).lower():
                print("  [X] ERROR: sqlite-vec extension is not available!")
                print("  Please install sqlite-vec and ensure it's accessible.")
                sys.exit(1)
            # If table already exists, that's fine
            if "already exists" not in str(e).lower():
                raise
            print("  [OK] vec0 virtual table already exists")
        
        # Create mapping table
        print("  Creating image_embeddings mapping table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                rowid INTEGER PRIMARY KEY,
                image_id INTEGER,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        print("  [OK] Mapping table ready")
        
        # Create binary_embeddings table for binary embeddings (for fast approximate search)
        print(f"  Creating binary_embeddings table (binary embedding dimension: {self.embedding_dim})...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS binary_embeddings (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_binary_embeddings_image_id 
                ON binary_embeddings(image_id)
            """)
            print("  [OK] binary_embeddings table ready")
        except Exception as e:
            print(f"  [WARNING] Could not create binary_embeddings: {e}")
        
        # Check existing image count
        cursor.execute("SELECT COUNT(*) FROM images")
        existing_count = cursor.fetchone()[0]
        if existing_count > 0:
            print(f"  Database contains {existing_count:,} existing images")
        
        conn.commit()
        conn.close()
        print(f"  [OK] Database initialized successfully")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _needs_thumbnail(self, file_path: str) -> bool:
        """Check if file needs a thumbnail (non-browser-supported formats)."""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in {'.pdf', '.tif', '.tiff', '.bmp'}
    
    def _get_thumbnail_path(self, file_path: str) -> str:
        """Get thumbnail path for a file."""
        file_hash = self._get_file_hash(file_path)
        file_ext = Path(file_path).suffix.lower()
        # Use hash to avoid path issues and ensure uniqueness
        thumbnail_filename = f"{file_hash}.jpg"
        thumbnail_dir = Path(DEFAULT_THUMBNAILS_DIR)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        return str(thumbnail_dir / thumbnail_filename)
    
    def _create_thumbnail(self, file_path: str, max_size: Tuple[int, int] = (400, 400)) -> Optional[str]:
        """Create thumbnail for a file. Returns thumbnail path if successful, None otherwise."""
        try:
            thumbnail_path = self._get_thumbnail_path(file_path)
            
            # Skip if thumbnail already exists
            if os.path.exists(thumbnail_path):
                return thumbnail_path
            
            # Load image
            image = self._load_image(file_path)
            if image is None:
                return None
            
            # Create thumbnail
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save as JPEG
            image.save(thumbnail_path, "JPEG", quality=85)
            return thumbnail_path
        except Exception as e:
            self._safe_print_path("Error creating thumbnail for ", file_path, e)
            return None
    
    def _safe_print_path(self, message: str, file_path: str, error: Exception = None) -> None:
        """Safely print a message with a file path that may contain Unicode characters."""
        try:
            if error:
                print(f"{message}{file_path}: {error}", flush=True)
            else:
                print(f"{message}{file_path}", flush=True)
        except UnicodeEncodeError:
            # Fallback: encode path as ASCII with error handling
            safe_path = file_path.encode('ascii', 'replace').decode('ascii')
            if error:
                print(f"{message}{safe_path}: {error}", flush=True)
            else:
                print(f"{message}{safe_path}", flush=True)
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from file, handling PDFs and regular images."""
        try:
            file_ext = Path(image_path).suffix.lower()
            if file_ext == '.pdf' and PDF_SUPPORT:
                # Convert first page of PDF to image using PyMuPDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(image_path)
                    if len(doc) == 0:
                        doc.close()
                        return None
                    # Get first page
                    page = doc[0]
                    # Render to image at 150 DPI (similar to pdf2image default)
                    mat = fitz.Matrix(150/72, 150/72)  # 72 is default DPI
                    pix = page.get_pixmap(matrix=mat)
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    doc.close()
                    return img
                except Exception as pdf_error:
                    # PDF conversion failed
                    self._safe_print_path("Error converting PDF ", image_path, pdf_error)
                    return None
            elif file_ext == '.pdf' and not PDF_SUPPORT:
                # PDF support not available
                self._safe_print_path("PDF support not available for ", image_path, None)
                return None
            else:
                return Image.open(image_path).convert("RGB")
        except Exception as e:
            self._safe_print_path("Error loading ", image_path, e)
            return None
    
    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding from an image file."""
        try:
            image = self._load_image(image_path)
            if image is None:
                return None
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # Convert inputs to match model dtype
            inputs = {k: v.to(self.dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Normalize for cosine similarity
                embedding = torch.nn.functional.normalize(outputs, p=2, dim=1)
                embedding = embedding.cpu().numpy().astype(np.float32).flatten()
            
            return embedding
        except Exception as e:
            self._safe_print_path("Error processing ", image_path, e)
            return None
    
    def _get_image_embeddings_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple images in a batch (much faster)."""
        images = []
        valid_paths = []
        
        # Load all images
        for image_path in image_paths:
            try:
                img = self._load_image(image_path)
                if img is not None:
                    images.append(img)
                    valid_paths.append(image_path)
            except Exception as e:
                self._safe_print_path("Error loading ", image_path, e)
                continue
        
        if not images:
            return [None] * len(image_paths)
        
        try:
            # Process all images in one batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            # Convert inputs to match model dtype
            inputs = {k: v.to(self.dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Normalize for cosine similarity
                embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
            
            # Map back to original paths (handle failed loads)
            result = [None] * len(image_paths)
            valid_idx = 0
            for i, path in enumerate(image_paths):
                if path in valid_paths:
                    result[i] = embeddings[valid_idx].flatten()
                    valid_idx += 1
            
            return result
        except Exception as e:
            print(f"Error processing batch: {e}")
            return [None] * len(image_paths)
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Extract embedding from text.
        
        CRITICAL: SigLIP 2 requires:
        1. Lowercase text (model was trained only on lowercase)
        2. Padding to exactly 64 tokens (max_length=64)
        3. Using get_text_features from SiglipModel to get projected vector
        """
        # 1. Lowercase is MANDATORY for SigLIP 2
        text = text.lower()
        
        # 2. Add the official prompt template
        prompt = f"this is a photo of {text}"
        
        # 3. Use processor with padding="max_length" and max_length=64 (strict requirement)
        inputs = self.processor(
            text=[prompt], 
            return_tensors="pt", 
            padding="max_length", 
            max_length=64
        ).to(self.device)
        
        # Convert inputs to match model dtype
        inputs = {k: v.to(self.dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use get_text_features to get the PROJECTED vector (aligned with images)
            text_features = self.model.get_text_features(**inputs)
            
            # 4. L2 Normalize
            embedding = text_features.cpu().numpy()[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
            embedding = embedding.astype(np.float32)
        
        return embedding
    
    def _apply_negative_embedding(self, embedding: np.ndarray, negative_emb: np.ndarray, 
                                   negative_weight: float, embedding1: np.ndarray, 
                                   embedding2: Optional[np.ndarray], weights: Tuple[float, float]) -> np.ndarray:
        """
        Apply negative embedding subtraction and re-normalize. Returns normalized embedding.
        
        Math: embedding = embedding - negative_weight * negative_emb, then normalize.
        For multiple negatives, this would be: embedding - w1*neg1 - w2*neg2 - ..., then normalize.
        """
        # Subtract negative embedding: move away from negative concept
        embedding = embedding - negative_weight * negative_emb
        # Re-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            print("Warning: Embedding became zero after negative subtraction, using original")
            # Restore original embedding
            if embedding2 is None:
                embedding = embedding1
            else:
                w1, w2 = weights[0] / (weights[0] + weights[1]), weights[1] / (weights[0] + weights[1])
                embedding = w1 * embedding1 + w2 * embedding2
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
        return embedding
    
    def _apply_multiple_negative_embeddings(self, embedding: np.ndarray, 
                                            negative_embs: List[np.ndarray],
                                            negative_weights: List[float],
                                            embedding1: np.ndarray,
                                            embedding2: Optional[np.ndarray],
                                            weights: Tuple[float, float]) -> np.ndarray:
        """
        Apply multiple negative embeddings with individual weights.
        
        Math: embedding = embedding - sum(weight_i * negative_emb_i), then normalize.
        This moves the embedding away from all negative concepts simultaneously.
        """
        # Subtract all negative embeddings
        for neg_emb, neg_weight in zip(negative_embs, negative_weights):
            embedding = embedding - neg_weight * neg_emb
        
        # Re-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            print("Warning: Embedding became zero after negative subtraction, using original")
            # Restore original embedding
            if embedding2 is None:
                embedding = embedding1
            else:
                w1, w2 = weights[0] / (weights[0] + weights[1]), weights[1] / (weights[0] + weights[1])
                embedding = w1 * embedding1 + w2 * embedding2
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
        return embedding
    
    def _sample_folder_sequences(self, files: List[Path]) -> List[Path]:
        """
        Sample likely frame sequences in a folder to avoid indexing thousands of near-identical frames.
        
        Heuristic:
        - Only consider sampling for large folders
        - Only sample when the folder name or the dominant filename prefix looks like a frame/render sequence
        - Avoid sampling common photo-camera prefixes (e.g. IMG_####, DSC_####)
        Returns the list of files to process (after sampling).
        """
        if len(files) <= 150:
            return files  # No sampling needed for small folders
        if not files:
            return files

        folder_name = files[0].parent.name.lower()
        folder_sequence_keywords = (
            "frame", "frames", "render", "renders", "sequence", "seq", "anim", "animation", "motion", "video"
        )
        folder_looks_like_sequence = any(k in folder_name for k in folder_sequence_keywords)
        
        # Extract numbers from filenames to detect sequences
        numbered_files: List[Tuple[int, Path, str]] = []
        for f in files:
            stem = f.stem
            # Try to extract trailing number (e.g., "frame_0001" -> 1, "render_042" -> 42)
            match = re.search(r'^(.*?)(\d+)$', stem)
            if match:
                prefix = (match.group(1) or "").lower()
                frame_num = int(match.group(2))
                numbered_files.append((frame_num, f, prefix))
        
        # Only sample if we found many numbered files and it looks like a frame/render sequence
        if len(numbered_files) > 150:
            # Identify dominant prefix among numbered files
            prefix_counts: Dict[str, int] = {}
            for _, _, pfx in numbered_files:
                prefix_counts[pfx] = prefix_counts.get(pfx, 0) + 1
            dominant_prefix, dominant_count = max(prefix_counts.items(), key=lambda kv: kv[1])
            dominant_frac = dominant_count / max(1, len(numbered_files))

            # Avoid sampling common camera/photo filename prefixes
            pfx_stripped = dominant_prefix.strip().strip("_- ")
            photo_prefixes = {
                "img", "dsc", "pict", "photo", "pxl", "mvimg", "dji", "gopr", "gopro", "scan"
            }
            dominant_is_photoish = (
                pfx_stripped in photo_prefixes
                or dominant_prefix.startswith(("img_", "dsc_", "pxl_", "mvimg_", "dji_", "gopr_"))
            )

            prefix_sequence_keywords = ("frame", "render", "shot", "output", "seq", "sequence", "anim", "animation")
            prefix_looks_like_sequence = any(k in dominant_prefix for k in prefix_sequence_keywords)

            should_sample = (
                dominant_frac >= 0.8
                and (folder_looks_like_sequence or prefix_looks_like_sequence)
                and not dominant_is_photoish
            )

            if not should_sample:
                return files

            numbered_files.sort(key=lambda x: x[0])  # Sort by frame number
            
            # Keep only every 100th frame (1, 101, 201, 301, ...)
            frames_to_keep = set()
            for i in range(0, len(numbered_files), 100):
                frames_to_keep.add(numbered_files[i][1])
            
            # Return only the files to keep (numbered files that are sampled + non-numbered files)
            result = []
            numbered_set = {f for _, f, _ in numbered_files}
            for f in files:
                if f in numbered_set:
                    if f in frames_to_keep:
                        result.append(f)
                else:
                    # Non-numbered files are always kept
                    result.append(f)
            
            return result
        
        # No sampling needed
        return files
    
    def _batch_check_processed(self, cursor, file_metadata: List[Tuple[str, float]]) -> set:
        """Batch check which images are already processed. Returns set of processed file paths."""
        if not file_metadata:
            return set()
        
        # SQLite has a limit of ~999 variables, so chunk into batches of 400 (200 pairs)
        processed = set()
        chunk_size = 400  # 200 file_path + last_modified pairs
        
        for i in range(0, len(file_metadata), chunk_size):
            chunk = file_metadata[i:i + chunk_size]
            placeholders = ','.join(['(?, ?)'] * len(chunk))
            values = [item for pair in chunk for item in pair]
            
            # Check if image exists AND has an embedding (either full or binary)
            # Check for full embeddings first, then binary embeddings
            cursor.execute(f"""
                SELECT i.file_path 
                FROM images i
                WHERE (i.file_path, i.last_modified) IN (VALUES {placeholders})
                AND (
                    EXISTS (SELECT 1 FROM image_embeddings ie WHERE ie.image_id = i.id)
                    OR EXISTS (SELECT 1 FROM binary_embeddings be WHERE be.image_id = i.id)
                )
            """, values)
            
            processed.update(row[0] for row in cursor.fetchall())
        
        return processed
    
    def scan_directory(self, root_dir: str, batch_size: int = 75, inference_batch_size: int = 16, profile: bool = False, limit: int = None, exclude_paths: List[str] = None, save_full_embeddings: bool = True):
        """
        Scan directory and process images.
        Automatically samples large sequences: if a folder has >100 numbered frames, only every 100th frame is indexed.
        
        Args:
            root_dir: Root directory to scan
            batch_size: Number of images to process before committing to DB (default: 75)
            inference_batch_size: Number of images to process in parallel for model inference (default: 16)
            profile: If True, print timing information for each step
            limit: Limit number of images to process (for testing, None = no limit)
            exclude_paths: List of directory paths to exclude from scanning
            save_full_embeddings: If True, save full embeddings to vec0 (default: True). Set to False for binary-only mode.
        """
        print("="*60)
        print("Starting Directory Scan")
        print("="*60)
        print(f"Root directory: {root_dir}")
        print(f"Database: {self.db_path}")
        print(f"Batch size (DB commits): {batch_size}")
        print(f"Inference batch size: {inference_batch_size}")
        if save_full_embeddings:
            print(f"Embedding mode: Full embeddings (vec0) + Binary embeddings")
        else:
            print(f"Embedding mode: Binary embeddings only (space-efficient mode)")
        if limit:
            print(f"Limit: {limit} images (testing mode)")
        print("="*60 + "\n")
        
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"[X] Error: Directory {root_dir} does not exist")
            return
        
        # Normalize exclude paths to absolute paths for comparison
        exclude_abs_paths = []
        if exclude_paths:
            for excl_path in exclude_paths:
                excl_abs = os.path.abspath(excl_path)
                exclude_abs_paths.append(excl_abs)
            print(f"Excluding {len(exclude_paths)} directory path(s):")
            for excl_path in exclude_paths:
                print(f"  - {excl_path}")
        
        # Supported image extensions (including PDFs)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        if PDF_SUPPORT:
            image_extensions.add('.pdf')
        print(f"Supported image formats: {', '.join(sorted(image_extensions))}")
        
        # First pass: count total files
        print("\n[Step 1/4] Counting image files...", flush=True)
        print("  Using fast directory traversal (os.walk)...", flush=True)
        # Use set to deduplicate (Windows is case-insensitive, so *.jpg and *.JPG find same files)
        # Convert extensions to lowercase for case-insensitive matching
        image_extensions_lower = {ext.lower() for ext in image_extensions}
        image_files_set = set()
        excluded_count = 0
        file_count = 0
        last_report = 0
        report_interval = 50000  # Report every 50k files
        
        # Use os.walk() which is much faster than rglob() on Windows
        # Traverse directory tree once and filter by extension
        root_str = str(root_path.absolute())
        for root, dirs, files in os.walk(root_str):
            # Check if current directory or any parent should be excluded
            root_abs = os.path.abspath(root)
            root_abs_norm = root_abs.lower()  # Case-insensitive comparison on Windows
            
            should_skip = False
            if exclude_abs_paths:
                for excl_abs in exclude_abs_paths:
                    excl_abs_norm = excl_abs.lower()
                    # Check if current directory is excluded, or is a subdirectory of excluded path
                    if root_abs_norm == excl_abs_norm or root_abs_norm.startswith(excl_abs_norm + os.sep):
                        # Skip this directory and all subdirectories
                        dirs[:] = []  # Clear dirs list to prevent descending
                        should_skip = True
                        excluded_count += 1
                        break
            
            if should_skip:
                continue
            
            for file in files:
                # Skip macOS resource fork files (._filename)
                if file.startswith('._'):
                    continue
                
                # Get file extension (case-insensitive)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions_lower:
                    # Build full path
                    full_path = os.path.join(root, file)
                    # Use absolute path for deduplication
                    abs_path = os.path.abspath(full_path)
                    image_files_set.add(abs_path)
                    
                    file_count = len(image_files_set)
                    if file_count - last_report >= report_interval:
                        print(f"  Found {file_count:,} unique image files so far...", flush=True)
                        last_report = file_count
        
        total_found = len(image_files_set)
        if excluded_count > 0:
            print(f"  Excluded {excluded_count:,} directories", flush=True)
        print(f"  Found {total_found:,} total image files", flush=True)
        
        image_files = [Path(p) for p in image_files_set]  # Convert back to list of Path objects
        
        # Group files by parent directory - we'll process folders one at a time
        files_by_dir = {}
        
        for img_file in image_files:
            parent = img_file.parent
            if parent not in files_by_dir:
                files_by_dir[parent] = []
            files_by_dir[parent].append(img_file)
        
        total_dirs = len(files_by_dir)
        print(f"  Grouped into {total_dirs:,} directories", flush=True)
        
        if total_dirs == 0:
            print("\n[X] No image files found!")
            return
        
        print("\n[Step 2/4] Connecting to database...", flush=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)  # Increase timeout for locked database
        # Enable WAL mode for better concurrent access (allows reads during writes)
        conn.execute("PRAGMA journal_mode=WAL")
        # Ensure extension is loaded for this connection
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        except:
            pass  # Already loaded or not needed
        
        cursor = conn.cursor()
        
        processed = 0
        skipped = 0
        errors = 0
        db_batch = []
        inference_batch = []
        inference_metadata = []
        
        # Profiling timers
        timers = defaultdict(float)
        timer_counts = defaultdict(int)
        
        try:
            # Process folders one at a time (enables folder-level resume)
            print("\n[Step 3/4] Processing images...", flush=True)
            
            # Estimate total files for progress bar (will be updated as we sample)
            estimated_total = total_found
            total_processed_estimate = 0
            
            # Sort folders for consistent processing order
            sorted_folders = sorted(files_by_dir.items(), key=lambda x: str(x[0]))
            total_folders = len(sorted_folders)
            
            print(f"  Processing {total_folders:,} folders...", flush=True)
            
            with tqdm(total=estimated_total, desc="Processing images", unit="img", unit_scale=True) as pbar:
                folder_num = 0
                sampled_folders = 0
                total_files_removed = 0
                
                for parent_dir, folder_files in sorted_folders:
                    try:
                        folder_num += 1
                        
                        # Apply sequence sampling to this folder
                        files_to_process = self._sample_folder_sequences(folder_files)
                        
                        if len(files_to_process) < len(folder_files):
                            # Sequence sampling was applied
                            removed = len(folder_files) - len(files_to_process)
                            total_files_removed += removed
                            sampled_folders += 1
                            # Update progress bar total to exclude sampled files
                            pbar.total = max(pbar.total - removed, pbar.n)
                        
                        # Check which files in this folder are already processed
                        folder_metadata = []
                        for img_path in files_to_process:
                            file_path = str(img_path.absolute())
                            last_modified = os.path.getmtime(file_path)
                            folder_metadata.append((file_path, last_modified))
                        
                        # Batch check for this folder
                        check_start = time.time()
                        processed_files = self._batch_check_processed(cursor, folder_metadata)
                        timers['check_db'] += time.time() - check_start
                        timer_counts['check_db'] += 1
                        
                        # Collect files to process from this folder
                        folder_to_process = []
                        for file_path, last_modified in folder_metadata:
                            if file_path not in processed_files:
                                folder_to_process.append((file_path, last_modified))
                            else:
                                skipped += 1
                                pbar.update(1)  # Update progress for skipped files
                        
                        # Apply limit if specified (for testing)
                        if limit is not None:
                            remaining_limit = limit - total_processed_estimate
                            if remaining_limit <= 0:
                                break  # Hit the limit
                            if len(folder_to_process) > remaining_limit:
                                folder_to_process = folder_to_process[:remaining_limit]
                        
                        # Process files from this folder
                        for file_path, last_modified in folder_to_process:
                            inference_batch.append(file_path)
                            inference_metadata.append((file_path, last_modified))
                            total_processed_estimate += 1
                            
                            # Check limit during processing
                            # (if limit reached, current batch will be processed below)
                            
                            # Process batch when full
                            if len(inference_batch) >= inference_batch_size:
                                # Get embeddings in batch
                                embed_start = time.time()
                                embeddings = self._get_image_embeddings_batch(inference_batch)
                                timers['inference'] += time.time() - embed_start
                                timer_counts['inference'] += len(inference_batch)
                                
                                # Hash files (can be parallelized but keeping simple for now)
                                hash_start = time.time()
                                for i, (file_path, last_modified) in enumerate(inference_metadata):
                                    if embeddings[i] is not None:
                                        file_hash = self._get_file_hash(file_path)
                                        db_batch.append((file_path, last_modified, file_hash, embeddings[i]))
                                    else:
                                        errors += 1
                                timers['hashing'] += time.time() - hash_start
                                timer_counts['hashing'] += len(inference_metadata)
                                
                                # Commit to database
                                if len(db_batch) >= batch_size:
                                    db_start = time.time()
                                    self._commit_with_retry(cursor, conn, db_batch, save_full_embeddings)
                                    timers['db_write'] += time.time() - db_start
                                    timer_counts['db_write'] += len(db_batch)
                                    processed += len(db_batch)
                                    db_batch = []
                                
                                pbar.update(len(inference_batch))
                                inference_batch = []
                                inference_metadata = []
                                
                                # Check if we hit the limit
                                if limit is not None and total_processed_estimate >= limit:
                                    break
                        
                        # Check limit after processing folder
                        if limit is not None and total_processed_estimate >= limit:
                            break
                    except Exception as e:
                        # Log error but continue with next folder
                        try:
                            folder_str = str(parent_dir)[-80:]
                        except:
                            folder_str = "unknown"
                        print(f"\n  [ERROR] Error processing folder {folder_num}/{total_folders}: {folder_str}", flush=True)
                        print(f"  Error: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        errors += len(folder_files)  # Count all files in folder as errors
                        pbar.update(len(folder_files))  # Update progress bar
                        continue  # Continue with next folder
                
                # Process remaining inference batch
                if inference_batch:
                    embed_start = time.time()
                    embeddings = self._get_image_embeddings_batch(inference_batch)
                    timers['inference'] += time.time() - embed_start
                    timer_counts['inference'] += len(inference_batch)
                    
                    hash_start = time.time()
                    for i, (file_path, last_modified) in enumerate(inference_metadata):
                        if embeddings[i] is not None:
                            file_hash = self._get_file_hash(file_path)
                            db_batch.append((file_path, last_modified, file_hash, embeddings[i]))
                        else:
                            errors += 1
                    timers['hashing'] += time.time() - hash_start
                    timer_counts['hashing'] += len(inference_metadata)
                    
                    pbar.update(len(inference_batch))
                
                # Commit remaining items
                if db_batch:
                    db_start = time.time()
                    self._commit_with_retry(cursor, conn, db_batch, save_full_embeddings)
                    timers['db_write'] += time.time() - db_start
                    timer_counts['db_write'] += len(db_batch)
                    processed += len(db_batch)
                
                # Final summary
                print(f"\n  Processed {folder_num:,} / {total_folders:,} folders", flush=True)
                if sampled_folders > 0:
                    print(f"  Sequence sampling: {sampled_folders} folders sampled, {total_files_removed:,} files removed (kept every 100th frame)", flush=True)
                
                # Apply limit if specified (for testing) - check if we hit the limit
                if limit is not None and total_processed_estimate >= limit:
                    print(f"  Limited to {limit} images for testing - stopping", flush=True)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Committing current batch...")
            # Commit any pending inference batch first
            if inference_batch:
                embed_start = time.time()
                embeddings = self._get_image_embeddings_batch(inference_batch)
                for i, (file_path, last_modified) in enumerate(inference_metadata):
                    if embeddings[i] is not None:
                        file_hash = self._get_file_hash(file_path)
                        db_batch.append((file_path, last_modified, file_hash, embeddings[i]))
                    else:
                        errors += 1
            
            # Commit database batch
            if db_batch:
                self._commit_with_retry(cursor, conn, db_batch, save_full_embeddings)
                processed += len(db_batch)
            print(f"Progress saved: {processed} processed, {skipped} skipped, {errors} errors")
            print("You can resume by running the same command - already processed images will be skipped.")
        
        finally:
            conn.close()
        
        print("\n[Step 4/4] Finalizing...", flush=True)
        print("="*60)
        print("Scan Complete!")
        print("="*60)
        print(f"  Processed: {processed:,} images")
        print(f"  Skipped: {skipped:,} images (already in database)")
        if errors > 0:
            print(f"  Errors: {errors:,} images")
        else:
            print(f"  Errors: 0")
        
        # Print profiling information
        if profile and timers:
            print(f"\n=== Performance Profile ===")
            total_time = sum(timers.values())
            for operation, total in timers.items():
                count = timer_counts.get(operation, 1)
                avg = total / count if count > 0 else 0
                pct = (total / total_time * 100) if total_time > 0 else 0
                print(f"  {operation:15s}: {total:8.2f}s total, {avg*1000:6.1f}ms avg, {pct:5.1f}% of time ({count} ops)")
            print(f"  {'TOTAL':15s}: {total_time:8.2f}s")
            if processed > 0:
                print(f"  Throughput: {processed/total_time:.1f} images/second")
        
        print("="*60 + "\n")
    
    def _commit_with_retry(self, cursor, conn, db_batch: List[Tuple], save_full_embeddings: bool, max_retries: int = 5):
        """Commit batch to database with retry logic for handling database locks."""
        for attempt in range(max_retries):
            try:
                self._commit_batch(cursor, db_batch, save_full_embeddings)
                conn.commit()
                return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
    
    def _commit_batch(self, cursor, batch: List[Tuple], save_full_embeddings: bool = False):
        """Commit a batch of images and embeddings to the database.
        
        Args:
            cursor: Database cursor
            batch: List of (file_path, last_modified, file_hash, embedding) tuples
            save_full_embeddings: If True, save full embeddings to vec0. If False, only save binary embeddings.
        """
        for file_path, last_modified, file_hash, embedding in batch:
            try:
                # Check if image already exists with matching timestamp (already processed)
                cursor.execute("""
                    SELECT id FROM images 
                    WHERE file_path = ? AND last_modified = ?
                """, (file_path, last_modified))
                existing_row = cursor.fetchone()
                
                if existing_row:
                    image_id = existing_row[0]
                    # Check if embedding already exists - if so, skip (already processed)
                    if save_full_embeddings:
                        cursor.execute("""
                            SELECT 1 FROM image_embeddings WHERE image_id = ?
                        """, (image_id,))
                        if cursor.fetchone():
                            continue  # Already has full embedding, skip
                    else:
                        # Binary-only mode: check if binary embedding exists
                        cursor.execute("""
                            SELECT 1 FROM binary_embeddings WHERE image_id = ?
                        """, (image_id,))
                        if cursor.fetchone():
                            continue  # Already has binary embedding, skip
                
                # Create thumbnail if needed (for PDF, TIF, BMP)
                if self._needs_thumbnail(file_path):
                    self._create_thumbnail(file_path)
                
                # Insert or update image metadata and get ID
                cursor.execute("""
                    INSERT OR REPLACE INTO images (file_path, last_modified, file_hash)
                    VALUES (?, ?, ?)
                """, (file_path, last_modified, file_hash))
                
                # Get image_id (handle INSERT OR REPLACE case)
                if cursor.lastrowid == 0:
                    cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
                    row = cursor.fetchone()
                    image_id = row[0] if row else None
                else:
                    image_id = cursor.lastrowid
                
                if image_id is None:
                    continue
                
                # Save full embeddings if requested
                if save_full_embeddings:
                    # Check if embedding already exists for this image
                    cursor.execute("""
                        SELECT rowid FROM image_embeddings WHERE image_id = ?
                    """, (image_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing embedding
                        vec_rowid = existing[0]
                        serialized_embedding = sqlite_vec.serialize_float32(embedding)
                        cursor.execute("""
                            UPDATE vec0 SET embedding = ? WHERE rowid = ?
                        """, (serialized_embedding, vec_rowid))
                    else:
                        # Insert new embedding
                        serialized_embedding = sqlite_vec.serialize_float32(embedding)
                        cursor.execute("""
                            INSERT INTO vec0 (embedding) VALUES (?)
                        """, (serialized_embedding,))
                        
                        vec_rowid = cursor.lastrowid
                        
                        # Link embedding to image
                        cursor.execute("""
                            INSERT INTO image_embeddings (rowid, image_id)
                            VALUES (?, ?)
                        """, (vec_rowid, image_id))
                
                # Always save binary embedding (only if it doesn't exist)
                cursor.execute("""
                    SELECT 1 FROM binary_embeddings WHERE image_id = ?
                """, (image_id,))
                if not cursor.fetchone():
                    # Binary embedding doesn't exist, insert it
                    binary_embedding = (embedding >= 0).astype(np.uint8)
                    binary_blob = binary_embedding.tobytes()
                    try:
                        cursor.execute("""
                            INSERT INTO binary_embeddings (image_id, embedding)
                            VALUES (?, ?)
                        """, (image_id, binary_blob))
                    except sqlite3.OperationalError:
                        # Binary table might not exist yet, skip silently
                        pass
            
            except sqlite3.IntegrityError:
                # Skip duplicates
                continue
            except Exception as e:
                self._safe_print_path("Error committing ", file_path, e)
                continue
    
    def _filter_duplicates(self, results: List[Tuple[str, float]], tolerance_bits: int = 2) -> List[Tuple[str, float]]:
        """
        Filter out duplicate images from search results based on binary embeddings.
        Keeps the result with highest similarity score when duplicates are found.
        
        Args:
            results: List of (file_path, similarity) tuples
            tolerance_bits: Maximum number of bits that can differ to still be considered duplicate (default: 2)
        
        Returns:
            Filtered list with duplicates removed
        """
        if len(results) == 0:
            return results
        
        # Get image IDs for all results
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        except:
            pass
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        
        # Map file paths to image IDs and binary embeddings
        file_to_id = {}
        id_to_binary = {}
        
        for file_path, similarity in results:
            cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                file_to_id[file_path] = row[0]
        
        # Fetch binary embeddings for all image IDs
        if file_to_id:
            image_ids = list(file_to_id.values())
            placeholders = ','.join(['?'] * len(image_ids))
            cursor.execute(f"""
                SELECT image_id, embedding 
                FROM binary_embeddings 
                WHERE image_id IN ({placeholders})
            """, image_ids)
            
            for image_id, binary_blob in cursor.fetchall():
                id_to_binary[image_id] = np.frombuffer(binary_blob, dtype=np.uint8)
        
        conn.close()
        
        # Find duplicates by comparing binary embeddings
        seen_embeddings = {}  # binary_embedding_tuple -> (file_path, similarity)
        filtered_results = []
        duplicates_removed = 0
        
        for file_path, similarity in results:
            if file_path not in file_to_id:
                # No embedding found, keep it
                filtered_results.append((file_path, similarity))
                continue
            
            image_id = file_to_id[file_path]
            if image_id not in id_to_binary:
                # No binary embedding, keep it
                filtered_results.append((file_path, similarity))
                continue
            
            binary_emb = id_to_binary[image_id]
            
            # Check if this is a duplicate of something we've already seen
            is_duplicate = False
            for seen_binary_tuple, (seen_path, seen_sim) in seen_embeddings.items():
                seen_binary = np.array(seen_binary_tuple, dtype=np.uint8)
                # Calculate Hamming distance (number of differing bits)
                diff_bits = np.sum(binary_emb != seen_binary)
                if diff_bits <= tolerance_bits:
                    # This is a duplicate - keep the one with higher similarity
                    is_duplicate = True
                    if similarity > seen_sim:
                        # Replace the seen one with this better match
                        seen_embeddings[seen_binary_tuple] = (file_path, similarity)
                        # Remove old one and add new one
                        filtered_results = [(fp, sim) for fp, sim in filtered_results if fp != seen_path]
                        filtered_results.append((file_path, similarity))
                    else:
                        # Keep the seen one, skip this
                        duplicates_removed += 1
                    break
            
            if not is_duplicate:
                # Not a duplicate, add it
                seen_embeddings[tuple(binary_emb)] = (file_path, similarity)
                filtered_results.append((file_path, similarity))
        
        if duplicates_removed > 0:
            print(f"Filtered out {duplicates_removed} duplicate(s) (tolerance: {tolerance_bits} bits)")
        
        # Sort by similarity again (in case we reordered)
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results
    
    def search(self, query: str, k: int = 10, is_image_path: bool = False, 
               query2: str = None, is_image_path2: bool = False, 
               weights: Tuple[float, float] = (0.5, 0.5),
               negative_query: str = None, negative_is_image: bool = False,
               negative_weight: float = 0.5,
               negative_queries: List[str] = None, negative_is_images: List[bool] = None,
               negative_weights: List[float] = None,
               filter_folders: List[str] = None,
               profile: bool = False,
               show_duplicates: bool = False) -> List[Tuple[str, float]]:
        """
        Search for similar images. Supports combined queries and negative prompts.
        
        Args:
            query: Text string or image file path (first query)
            k: Number of results to return
            is_image_path: If True, treat query as image path; otherwise as text
            query2: Optional second query (text or image path) for combined positive search
            is_image_path2: If True, treat query2 as image path; otherwise as text
            weights: Tuple of (weight1, weight2) for combining positive queries. Default (0.5, 0.5)
            negative_query: Optional negative prompt (text or image path) to exclude
            negative_is_image: If True, treat negative_query as image path; otherwise as text
            negative_weight: Weight for negative prompt subtraction (default: 0.5)
            filter_folders: Optional list of folder paths to filter results (only show images in these folders)
            profile: If True, print timing information for each step
            show_duplicates: If True, show duplicate images in results. If False (default), filter out duplicates based on binary embeddings (tolerance: 2 bits)
        
        Returns:
            List of (file_path, similarity_score) tuples
        """
        timings = {}
        
        # Get first query embedding
        if is_image_path:
            if not os.path.exists(query):
                print(f"Error: Image file {query} does not exist")
                return []
            print(f"Processing image query: {query}")
            start = time.time()
            embedding1 = self._get_image_embedding(query)
            timings['embedding1_image'] = time.time() - start
            if embedding1 is None:
                print("Error: Failed to generate embedding from image")
                return []
        else:
            print(f"Processing text query: {query}")
            start = time.time()
            embedding1 = self._get_text_embedding(query)
            timings['embedding1_text'] = time.time() - start
        
        # If second query provided, combine embeddings
        if query2 is not None:
            # Get second query embedding
            if is_image_path2:
                if not os.path.exists(query2):
                    print(f"Error: Image file {query2} does not exist")
                    return []
                print(f"Processing second image query: {query2}")
                start = time.time()
                embedding2 = self._get_image_embedding(query2)
                timings['embedding2_image'] = time.time() - start
                if embedding2 is None:
                    print("Error: Failed to generate embedding from second image")
                    return []
            else:
                print(f"Processing second text query: {query2}")
                start = time.time()
                embedding2 = self._get_text_embedding(query2)
                timings['embedding2_text'] = time.time() - start
            
            # Normalize weights
            total_weight = weights[0] + weights[1]
            if total_weight == 0:
                weights = (0.5, 0.5)
                total_weight = 1.0
            w1, w2 = weights[0] / total_weight, weights[1] / total_weight
            
            # Combine embeddings with weighted average
            start = time.time()
            embedding = w1 * embedding1 + w2 * embedding2
            
            # L2 normalize the combined embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                print("Warning: Combined embedding has zero norm, using first query only")
                embedding = embedding1
            timings['combine_embeddings'] = time.time() - start
        else:
            embedding = embedding1
        
        # Apply negative prompt(s) if provided (subtract negative embedding(s))
        # Support both single negative (backward compatible) and multiple negatives
        negative_embs_list = []
        negative_weights_list = []
        
        # Handle single negative (backward compatible)
        if negative_query is not None:
            if negative_is_image:
                if not os.path.exists(negative_query):
                    print(f"Warning: Negative image file {negative_query} does not exist, ignoring negative prompt")
                else:
                    print(f"Processing negative image: {negative_query}")
                    start = time.time()
                    negative_emb = self._get_image_embedding(negative_query)
                    timings['negative_embedding_image'] = time.time() - start
                    if negative_emb is not None:
                        negative_embs_list.append(negative_emb)
                        negative_weights_list.append(negative_weight)
            else:
                print(f"Processing negative text: {negative_query}")
                start = time.time()
                negative_emb = self._get_text_embedding(negative_query)
                timings['negative_embedding_text'] = time.time() - start
                if negative_emb is not None:
                    negative_embs_list.append(negative_emb)
                    negative_weights_list.append(negative_weight)
        
        # Handle multiple negatives
        if negative_queries is not None:
            for i, neg_q in enumerate(negative_queries):
                neg_is_img = negative_is_images[i] if negative_is_images and i < len(negative_is_images) else False
                neg_w = negative_weights[i] if negative_weights and i < len(negative_weights) else negative_weight
                
                if neg_is_img:
                    if not os.path.exists(neg_q):
                        print(f"Warning: Negative image file {neg_q} does not exist, skipping")
                        continue
                    print(f"Processing negative image {i+1}: {neg_q}")
                    start = time.time()
                    neg_emb = self._get_image_embedding(neg_q)
                    timings[f'negative_embedding_image_{i}'] = time.time() - start
                    if neg_emb is not None:
                        negative_embs_list.append(neg_emb)
                        negative_weights_list.append(neg_w)
                else:
                    print(f"Processing negative text {i+1}: {neg_q}")
                    start = time.time()
                    neg_emb = self._get_text_embedding(neg_q)
                    timings[f'negative_embedding_text_{i}'] = time.time() - start
                    if neg_emb is not None:
                        negative_embs_list.append(neg_emb)
                        negative_weights_list.append(neg_w)
        
        # Apply all negative embeddings
        if negative_embs_list:
            if len(negative_embs_list) == 1:
                # Single negative - use original method
                print(f"Applying negative prompt (weight: {negative_weights_list[0]})...")
                start = time.time()
                embedding = self._apply_negative_embedding(
                    embedding, negative_embs_list[0], negative_weights_list[0], embedding1,
                    embedding2 if query2 is not None else None, weights
                )
                timings['apply_negative'] = time.time() - start
            else:
                # Multiple negatives - use new method
                print(f"Applying {len(negative_embs_list)} negative prompts (weights: {', '.join([f'{w:.2f}' for w in negative_weights_list])})...")
                start = time.time()
                embedding = self._apply_multiple_negative_embeddings(
                    embedding, negative_embs_list, negative_weights_list, embedding1,
                    embedding2 if query2 is not None else None, weights
                )
                timings['apply_negative'] = time.time() - start
        
        start = time.time()
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable WAL mode for better concurrent access (allows reads during writes)
        conn.execute("PRAGMA journal_mode=WAL")
        # Ensure extension is loaded for this connection
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        except Exception as e:
            print(f"Warning: Could not load sqlite-vec in search: {e}")
        timings['db_connect'] = time.time() - start
        
        cursor = conn.cursor()
        
        # Check if binary embeddings exist
        try:
            cursor.execute("SELECT COUNT(*) FROM binary_embeddings")
            binary_count = cursor.fetchone()[0]
            if binary_count == 0:
                print("Error: Database has no embeddings. Please run scan first.")
                conn.close()
                return []
        except sqlite3.OperationalError as e:
            print(f"Error: binary_embeddings table not accessible: {e}")
            print("Please run scan first to create embeddings.")
            conn.close()
            return []
        
        print(f"Searching database for top {k} results...")
        if filter_folders:
            print(f"Filtering to {len(filter_folders)} folder(s):")
            for folder in filter_folders:
                print(f"  - {folder}")
        
        try:
            # Build WHERE clause for folder filtering
            start = time.time()
            where_clause = ""
            
            if filter_folders:
                # Normalize folder paths (ensure they end with path separator for prefix matching)
                normalized_folders = []
                for folder in filter_folders:
                    folder_abs = os.path.abspath(folder)
                    # Ensure folder path ends with separator for prefix matching
                    if not folder_abs.endswith(os.sep):
                        folder_abs += os.sep
                    normalized_folders.append(folder_abs)
                
                # Create WHERE clause with LIKE conditions for each folder
                # Use OR to match any of the folders
                folder_conditions = []
                for folder_path in normalized_folders:
                    folder_conditions.append("i.file_path LIKE ? ESCAPE '\\'")
                
                where_clause = "WHERE (" + " OR ".join(folder_conditions) + ")"
            timings['build_query'] = time.time() - start
            
            # Check if full embeddings are available (preferred for accuracy)
            use_full_embeddings = False
            try:
                cursor.execute("SELECT COUNT(*) FROM vec0")
                vec0_count = cursor.fetchone()[0]
                if vec0_count > 0:
                    use_full_embeddings = True
            except sqlite3.OperationalError:
                pass  # vec0 table doesn't exist
            
            # Check if binary embeddings are available (fallback)
            use_binary = False
            try:
                cursor.execute("SELECT COUNT(*) FROM binary_embeddings")
                binary_count = cursor.fetchone()[0]
                if binary_count > 0:
                    use_binary = True
            except sqlite3.OperationalError:
                pass  # Binary table doesn't exist
            
            if not use_full_embeddings and not use_binary:
                print("Error: No embeddings found in database. Please run scan first.")
                conn.close()
                return []
            
            start = time.time()
            
            if use_full_embeddings:
                # Use full embeddings for accurate search (if available)
                query_vec = sqlite_vec.serialize_float32(embedding)
                
                # Use sqlite-vec's cosine distance for accurate search
                search_sql = f"""
                    SELECT 
                        i.file_path,
                        vec_distance_cosine(vec0.embedding, ?) as distance
                    FROM vec0
                    JOIN image_embeddings ie ON vec0.rowid = ie.rowid
                    JOIN images i ON ie.image_id = i.id
                    {where_clause}
                    ORDER BY distance ASC
                    LIMIT ?
                """
                search_params = [query_vec]
                if filter_folders and normalized_folders:
                    for folder_path in normalized_folders:
                        escaped_path = folder_path.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                        search_params.append(escaped_path + '%')
                search_params.append(k)
                
                cursor.execute(search_sql, tuple(search_params))
                results_raw = cursor.fetchall()
                
                # Convert distance to similarity score
                top_results = []
                for file_path, distance in results_raw:
                    similarity = 1.0 - distance
                    top_results.append((file_path, similarity))
                
            else:
                # Fallback to binary embeddings
                # Convert query embedding to binary (sign-based quantization)
                query_binary = (embedding >= 0).astype(np.uint8)
                
                # Fetch binary embeddings and compute binary dot product
                search_sql = f"""
                    SELECT 
                        be.image_id,
                        be.embedding,
                        i.file_path
                    FROM binary_embeddings be
                    JOIN images i ON be.image_id = i.id
                    {where_clause}
                """
                search_params = []
                if filter_folders and normalized_folders:
                    for folder_path in normalized_folders:
                        escaped_path = folder_path.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                        search_params.append(escaped_path + '%')
                
                cursor.execute(search_sql, tuple(search_params))
                all_binary_embeddings = cursor.fetchall()
                
                # Compute binary dot product for all candidates
                candidate_scores = []
                for image_id, binary_blob, file_path in all_binary_embeddings:
                    # Unpack binary embedding from BLOB
                    candidate_binary = np.frombuffer(binary_blob, dtype=np.uint8)
                    # Binary dot product (number of matching bits)
                    binary_score = np.dot(query_binary, candidate_binary)
                    # Normalize to similarity score (0-1 range)
                    # Maximum possible score is embedding_dim (all bits match)
                    similarity = float(binary_score) / self.embedding_dim
                    candidate_scores.append((file_path, similarity))
                
                # Sort by similarity (descending) and take top k
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                top_results = candidate_scores[:k]
            
            timings['db_query'] = time.time() - start
            
            start = time.time()
            results = [(file_path, float(similarity)) for file_path, similarity in top_results]
            timings['process_results'] = time.time() - start
        
        except Exception as e:
            print(f"Error during search: {e}")
            conn.close()
            return []
        
        conn.close()
        
        # Filter duplicates if not showing them
        if not show_duplicates and len(results) > 0:
            results = self._filter_duplicates(results, tolerance_bits=2)
        
        # Print profiling information if requested
        if profile and timings:
            print(f"\n=== Search Performance Profile ===")
            total_time = sum(timings.values())
            for operation, duration in sorted(timings.items(), key=lambda x: x[1], reverse=True):
                pct = (duration / total_time * 100) if total_time > 0 else 0
                print(f"  {operation:25s}: {duration*1000:7.2f}ms ({pct:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total_time*1000:7.2f}ms")
            print("="*40 + "\n")
        
        return results
    
    def generate_html_gallery(self, results: List[Tuple[str, float]], output_file: str = "results.html", query: str = None):
        """Generate HTML gallery for search results.
        
        For non-browser-friendly formats (PDF/TIF/BMP), uses cached thumbnails. If missing, generates thumbnails
        on-demand for the search results only.
        """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Search Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .query {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            font-size: 16px;
        }}
        .query strong {{
            color: #1976D2;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .result-item {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .result-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .image-container {{
            width: 100%;
            max-height: 400px;
            overflow: hidden;
            border-radius: 4px;
            margin-bottom: 10px;
            background: #f0f0f0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 400px;
            width: auto;
            height: auto;
            object-fit: contain;
        }}
        .score {{
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 8px;
        }}
        .file-path {{
            font-size: 12px;
            color: #666;
            word-break: break-all;
            margin-top: 8px;
        }}
        .file-path strong {{
            color: #333;
        }}
        .actions {{
            margin-top: 8px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .actions a {{
            display: inline-block;
            padding: 6px 12px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            transition: background 0.2s;
        }}
        .actions a:hover {{
            background: #1976D2;
        }}
        .actions a.folder-link {{
            background: #4CAF50;
        }}
        .actions a.folder-link:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <h1>Image Search Results</h1>
    {f'<div class="query"><strong>Query:</strong> {query}</div>' if query else ''}
    <p>Found {len(results)} results</p>
    <div class="gallery">
"""
        
        for file_path, similarity in results:
            # Convert Windows path to localexplorer: URL format
            # Use the path as-is with backslashes for localexplorer protocol
            file_url = f"localexplorer:{file_path}"
            
            # Get folder path (directory containing the file)
            folder_path = str(Path(file_path).parent)
            folder_url = f"localexplorer:{folder_path}"
            
            # Check if we need to use thumbnail (for PDF, TIF, BMP)
            file_ext = Path(file_path).suffix.lower()
            needs_thumb = file_ext in {'.pdf', '.tif', '.tiff', '.bmp'}
            
            if needs_thumb:
                # Use thumbnail if it exists; otherwise generate one on-demand for search results.
                thumbnail_path = self._get_thumbnail_path(file_path)
                if not os.path.exists(thumbnail_path):
                    self._create_thumbnail(file_path)

                if os.path.exists(thumbnail_path):
                    # Convert thumbnail path to file:// URL
                    thumb_display_url = thumbnail_path.replace('\\', '/')
                    if thumb_display_url[1] == ':' and len(thumb_display_url) > 2:
                        thumb_display_url = f"file:///{thumb_display_url}"
                    elif thumb_display_url.startswith('/'):
                        thumb_display_url = f"file://{thumb_display_url}"
                    else:
                        thumb_display_url = f"file:///{thumb_display_url}"
                    file_display_url = thumb_display_url
                else:
                    # Thumbnail doesn't exist, use placeholder
                    file_display_url = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2RkZCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5UaHVtYm5haWwgbm90IGF2YWlsYWJsZTwvdGV4dD48L3N2Zz4="
            else:
                # Convert file path to file:// URL for image display
                # Standard format: file:///E:/path/to/file.jpg (3 slashes for Windows absolute paths)
                file_display_url = file_path.replace('\\', '/')
                # Ensure proper file:// URL format (always 3 slashes for absolute paths)
                if file_display_url[1] == ':' and len(file_display_url) > 2:
                    # Windows absolute path with drive letter: E:/path -> file:///E:/path
                    file_display_url = f"file:///{file_display_url}"
                elif file_display_url.startswith('/'):
                    # Already has leading slash: /path -> file:///path
                    file_display_url = f"file://{file_display_url}"
                else:
                    # Relative path: path -> file:///path
                    file_display_url = f"file:///{file_display_url}"
            
            # Get just the filename for display
            filename = Path(file_path).name
            
            html_content += f"""        <div class="result-item">
            <div class="image-container">
                <img src="{file_display_url}" alt="{filename}" loading="lazy" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2RkZCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+';">
            </div>
            <div class="score">Similarity: {similarity:.4f}</div>
            <div class="file-path">
                <strong>{filename}</strong><br>
                <small>{file_path}</small>
            </div>
            <div class="actions">
                <a href="{file_url}">Open Image</a>
                <a href="{folder_url}" class="folder-link">Open Folder</a>
            </div>
        </div>
"""
        
        html_content += """    </div>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML gallery saved to {output_file}")


def generate_output_filename(query: str, is_image_path: bool = False, results_dir: Path = None) -> str:
    """Generate a safe filename from query, with auto-incrementing if file exists."""
    if results_dir is None:
        results_dir = Path(DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if is_image_path:
        # Use the image filename (without extension) as the query name
        query_name = Path(query).stem
    else:
        # Sanitize text query for filename
        # Remove or replace invalid characters
        query_name = re.sub(r'[<>:"/\\|?*]', '_', query)
        # Replace spaces with underscores
        query_name = query_name.replace(' ', '_')
        # Limit length
        if len(query_name) > 100:
            query_name = query_name[:100]
        # Remove trailing dots/spaces
        query_name = query_name.rstrip('. ')
        # If empty after sanitization, use a default
        if not query_name:
            query_name = "query"
    
    # Generate base filename
    base_filename = f"{query_name}.html"
    output_file = results_dir / base_filename
    
    # If file exists, add _2, _3, etc.
    counter = 1
    while output_file.exists():
        counter += 1
        output_file = results_dir / f"{query_name}_{counter}.html"
    
    return str(output_file)


def main():
    # Flush stdout immediately to ensure messages appear
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description="Searchable Image Database using SigLIP 2")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    # Scan mode
    scan_parser = subparsers.add_parser('scan', help='Scan directory and process images')
    scan_parser.add_argument('directory', help='Root directory to scan')
    scan_parser.add_argument('--db', default=None, help='Database path (required unless using --db-name)')
    scan_parser.add_argument('--db-name', default=None, help=f"Database filename in {DEFAULT_DB_DIR} (e.g. products_database.db)")
    scan_parser.add_argument('--batch-size', type=int, default=75, help='Batch size for DB commits')
    scan_parser.add_argument('--inference-batch-size', type=int, default=16, help='Batch size for model inference (higher = faster but more VRAM)')
    scan_parser.add_argument('--profile', action='store_true', help='Show performance profiling information')
    scan_parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process (for testing)')
    scan_parser.add_argument('--model-cache', default=DEFAULT_MODEL_CACHE_DIR, help='Model cache directory')
    scan_parser.add_argument('--exclude', action='append', help='Exclude directory path (can be used multiple times, e.g., --exclude "/path/to/exclude")')
    scan_parser.add_argument('--binary-only', action='store_true', help='Only save binary embeddings (space-efficient mode, default: saves full embeddings)')
    
    # Search mode
    search_parser = subparsers.add_parser('search', help='Search for similar images')
    search_parser.add_argument('query', nargs='?', help='Text query or image file path (optional if using --interactive)')
    search_parser.add_argument('-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--image', action='store_true', help='Treat query as image file path')
    search_parser.add_argument('--query2', help='Second query for combined search (text or image path)')
    search_parser.add_argument('--image2', action='store_true', help='Treat query2 as image file path')
    search_parser.add_argument('--weights', nargs=2, type=float, default=[0.5, 0.5], metavar=('W1', 'W2'), help='Weights for combining queries (default: 0.5 0.5)')
    search_parser.add_argument('--negative', help='Negative prompt to exclude (text or image path, use --negative-image for image)')
    search_parser.add_argument('--negative-image', action='store_true', help='Treat negative prompt as image file path')
    search_parser.add_argument('--negative-weight', type=float, default=0.5, help='Weight for negative prompt subtraction (default: 0.5)')
    search_parser.add_argument('--db', default=None, help='Database path (required unless using --db-name)')
    search_parser.add_argument('--db-name', default=None, help=f"Database filename in {DEFAULT_DB_DIR} (e.g. photos_database.db)")
    search_parser.add_argument('--model-cache', default=DEFAULT_MODEL_CACHE_DIR, help='Model cache directory')
    search_parser.add_argument('--output', default='results.html', help='Output HTML file')
    search_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive session mode: load model once and run multiple queries (default when query provided)')
    search_parser.add_argument('--no-session', action='store_true', help='Exit after processing query instead of keeping session open')
    search_parser.add_argument('--folder', action='append', help='Filter results to only show images in this folder (can be used multiple times, e.g., --folder "/path/to/folder")')
    search_parser.add_argument('--profile', action='store_true', help='Show performance profiling information for search')
    search_parser.add_argument('--show-duplicates', action='store_true', help='Show duplicate images in results (default: duplicates are filtered out)')
    
    args = parser.parse_args()
    sys.stdout.flush()
    
    if args.mode == 'scan':
        print("Starting scan mode...\n", flush=True)
        # Require explicit DB selection to avoid accidental mixing
        try:
            db_path = resolve_db_path(args.db, getattr(args, "db_name", None), DEFAULT_DB_DIR)
        except ValueError:
            print("\n[X] Error: No database selected.")
            print("Please specify either:")
            print("  --db \"C:\\Image-database\\products_database.db\"")
            print("  --db-name products_database.db")
            print(f"\nDatabase directory: {DEFAULT_DB_DIR}")
            dbs = list_db_files(DEFAULT_DB_DIR)
            if dbs:
                print("Available .db files:")
                for name in dbs:
                    print(f"  - {name}")
            else:
                print("No .db files found in database directory.")
            sys.exit(2)
        # Use None for model_cache to use default location if not specified
        model_cache = args.model_cache if args.model_cache else None
        print("Initializing database connection and loading model...", flush=True)
        db = ImageDatabase(db_path, model_cache)
        print("\nStarting directory scan...\n")
        sys.stdout.flush()
        db.scan_directory(
            args.directory, 
            batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size,
            profile=args.profile,
            limit=args.limit,
            exclude_paths=args.exclude if args.exclude else None,
            save_full_embeddings=not args.binary_only
        )
    
    elif args.mode == 'search':
        print("Starting search mode...\n")
        # Require explicit DB selection to avoid accidental mixing
        try:
            db_path = resolve_db_path(args.db, getattr(args, "db_name", None), DEFAULT_DB_DIR)
        except ValueError:
            print("\n[X] Error: No database selected.")
            print("Please specify either:")
            print("  --db \"C:\\Image-database\\photos_database.db\"")
            print("  --db-name photos_database.db")
            print(f"\nDatabase directory: {DEFAULT_DB_DIR}")
            dbs = list_db_files(DEFAULT_DB_DIR)
            if dbs:
                print("Available .db files:")
                for name in dbs:
                    print(f"  - {name}")
            else:
                print("No .db files found in database directory.")
            sys.exit(2)
        
        # Check if database file exists before initializing
        if not os.path.exists(db_path):
            print(f"\n[X] Error: Database file does not exist: {db_path}")
            print(f"\nDatabase directory: {DEFAULT_DB_DIR}")
            dbs = list_db_files(DEFAULT_DB_DIR)
            if dbs:
                print("Available .db files:")
                for name in dbs:
                    print(f"  - {name}")
            else:
                print("No .db files found in database directory.")
            sys.exit(2)
        
        # Verify database has the expected schema (check for images table)
        try:
            conn_check = sqlite3.connect(db_path)
            cursor_check = conn_check.cursor()
            cursor_check.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
            if not cursor_check.fetchone():
                conn_check.close()
                print(f"\n[X] Error: Database file exists but does not contain the expected schema: {db_path}")
                print("The database appears to be empty or not a valid image database.")
                sys.exit(2)
            conn_check.close()
        except sqlite3.Error as e:
            print(f"\n[X] Error: Could not verify database schema: {e}")
            sys.exit(2)
        
        # Use None for model_cache to use default location if not specified
        model_cache = args.model_cache if args.model_cache else None
        print("Initializing database connection and loading model...")
        db = ImageDatabase(db_path, model_cache)
        
        # Default to interactive mode if query is provided (unless --no-session is used)
        use_session = args.interactive or (args.query is not None and not args.no_session)
        
        if use_session:
            # Interactive session mode
            print("\n" + "="*60)
            print("Interactive Search Session")
            print("="*60)
            if args.query:
                print("Processing initial query, then session will remain open for more queries...")
            else:
                print("Model loaded and ready! Enter queries below.")
            print("Commands:")
            print("  - Enter a text query to search")
            print("  - Type 'image:<path>' to search by image")
            print("  - Type 'image:<path1> + image:<path2>' for combined image search")
            print("  - Type 'image:<path> + <text>' or '<text> + image:<path>' for image+text search")
            print("  - Type '<query> - <negative>' to exclude concepts (e.g., 'colourful design - grey monochrome')")
            print("  - Type '<query> - <neg1> - <neg2>' for multiple negatives (e.g., 'design - grey - abstract')")
            print("  - Type 'k:<number>' to change number of results (default: 10)")
            print("  - Type 'folder:<path>' to filter results to a folder (can use multiple times)")
            print("  - Type 'folder:clear' to clear folder filters")
            print("  - Type 'duplicates:show' to show duplicate images (default: hidden)")
            print("  - Type 'duplicates:hide' to hide duplicate images (default)")
            print("  - Type 'quit' or 'exit' to end session")
            print("="*60 + "\n")
            
            current_k = args.k
            is_image_query = args.image
            query2 = args.query2
            is_image_query2 = args.image2
            weights = tuple(args.weights)
            filter_folders = args.folder if args.folder else []
            profile_enabled = args.profile
            show_duplicates = args.show_duplicates
            
            # Handle initial query parameters from command line
            initial_negative_query = args.negative
            initial_negative_is_image = args.negative_image
            initial_negative_weight = args.negative_weight
            initial_query2 = args.query2
            initial_is_image_query2 = args.image2
            initial_is_image_query = args.image
            
            # Check if stdin is available (interactive terminal)
            is_interactive = sys.stdin.isatty()
            
            while True:
                try:
                    if args.query:
                        # Use provided query on first iteration
                        query = args.query
                        args.query = None  # Clear for next iterations
                        # Use command line parameters for first query
                        negative_query = initial_negative_query
                        negative_is_image = initial_negative_is_image
                        negative_weight = initial_negative_weight
                        negative_queries = None
                        negative_is_images = None
                        negative_weights = None
                        query2 = initial_query2
                        is_image_query2 = initial_is_image_query2
                        is_image_query = initial_is_image_query
                        # Clear initial parameters after first use
                        initial_negative_query = None
                        initial_query2 = None
                    else:
                        # Only try to read from stdin if it's available
                        if not is_interactive:
                            break  # Exit if stdin is not available
                        query = input("Query> ").strip()
                        # Reset parameters for subsequent queries
                        negative_query = None
                        negative_is_image = False
                        negative_weight = 0.5
                        negative_queries = None
                        negative_is_images = None
                        negative_weights = None
                        query2 = None
                        is_image_query2 = False
                        is_image_query = False
                    
                    if not query:
                        if not is_interactive:
                            break  # Exit if no query and not interactive
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("Ending session. Goodbye!")
                        break
                    
                    if query.lower().startswith('k:'):
                        try:
                            current_k = int(query.split(':', 1)[1].strip())
                            print(f"Number of results set to {current_k}")
                            continue
                        except ValueError:
                            print("Invalid number. Usage: k:20")
                            continue
                    
                    if query.lower().startswith('folder:'):
                        folder_path = query.split(':', 1)[1].strip()
                        if folder_path.lower() == 'clear':
                            filter_folders = []
                            print("Folder filters cleared")
                        else:
                            folder_abs = os.path.abspath(folder_path)
                            if os.path.isdir(folder_abs):
                                if folder_abs not in filter_folders:
                                    filter_folders.append(folder_abs)
                                    print(f"Added folder filter: {folder_abs}")
                                else:
                                    print(f"Folder already in filter list: {folder_abs}")
                            else:
                                print(f"Warning: Folder does not exist: {folder_abs}")
                        if filter_folders:
                            print(f"Current folder filters ({len(filter_folders)}):")
                            for f in filter_folders:
                                print(f"  - {f}")
                        continue
                    
                    if query.lower().startswith('duplicates:'):
                        dup_setting = query.split(':', 1)[1].strip().lower()
                        if dup_setting == 'show':
                            show_duplicates = True
                            print("Duplicate images will be shown")
                        elif dup_setting == 'hide':
                            show_duplicates = False
                            print("Duplicate images will be hidden (default)")
                        else:
                            print("Invalid option. Use 'duplicates:show' or 'duplicates:hide'")
                        continue
                    
                    # Parse negative queries (format: "query - negative1 - negative2 ...") if not already set from command line
                    if negative_query is None and ' - ' in query:
                        parts = query.split(' - ', 1)
                        query = parts[0].strip()
                        negative_str = parts[1].strip()
                        
                        # Support multiple negatives separated by " - "
                        negative_parts = [p.strip() for p in negative_str.split(' - ')]
                        
                        if len(negative_parts) == 1:
                            # Single negative (backward compatible)
                            if negative_parts[0].lower().startswith('image:'):
                                negative_query = negative_parts[0].split(':', 1)[1].strip()
                                negative_is_image = True
                            else:
                                negative_query = negative_parts[0]
                                negative_is_image = False
                        else:
                            # Multiple negatives - parse each one
                            negative_queries_list = []
                            negative_is_images_list = []
                            for neg_part in negative_parts:
                                if neg_part.lower().startswith('image:'):
                                    neg_path = neg_part.split(':', 1)[1].strip()
                                    negative_queries_list.append(neg_path)
                                    negative_is_images_list.append(True)
                                else:
                                    negative_queries_list.append(neg_part)
                                    negative_is_images_list.append(False)
                            
                            # Store as multiple negatives
                            negative_queries = negative_queries_list
                            negative_is_images = negative_is_images_list
                            negative_weights = [negative_weight] * len(negative_queries_list)  # Use default weight for all
                            print(f"Parsed {len(negative_queries_list)} negative queries")
                    
                    # Parse combined queries (format: "query1 + query2") if not already set from command line
                    if query2 is None:
                        query_parts = [q.strip() for q in query.split('+', 1)]
                        if len(query_parts) == 2:
                            # Combined query
                            q1, q2 = query_parts
                            
                            # Parse first query
                            if q1.lower().startswith('image:'):
                                query = q1.split(':', 1)[1].strip()
                                is_image_query = True
                            else:
                                query = q1
                                is_image_query = False
                            
                            # Parse second query
                            if q2.lower().startswith('image:'):
                                query2 = q2.split(':', 1)[1].strip()
                                is_image_query2 = True
                            else:
                                query2 = q2
                                is_image_query2 = False
                            
                            print(f"\nCombined search:")
                            print(f"  Query 1: {query} ({'image' if is_image_query else 'text'})")
                            print(f"  Query 2: {query2} ({'image' if is_image_query2 else 'text'})")
                            print(f"  Weights: {weights[0]:.1f} / {weights[1]:.1f}")
                            if negative_queries:
                                print(f"  Negatives ({len(negative_queries)}): {', '.join(negative_queries)}")
                            elif negative_query:
                                print(f"  Negative: {negative_query} ({'image' if negative_is_image else 'text'})")
                            print(f"  Number of results: {current_k}")
                        else:
                            # Single query
                            query2 = None
                            if query.lower().startswith('image:'):
                                image_path = query.split(':', 1)[1].strip()
                                query = image_path
                                is_image_query = True
                            else:
                                is_image_query = False
                            
                            print(f"\nSearching for: {query}")
                            if negative_queries:
                                print(f"  Negatives ({len(negative_queries)}): {', '.join(negative_queries)}")
                            elif negative_query:
                                print(f"  Negative: {negative_query} ({'image' if negative_is_image else 'text'})")
                            print(f"  Number of results: {current_k}")
                    else:
                        # Using command line query2, just print info
                        print(f"\nCombined search:")
                        print(f"  Query 1: {query} ({'image' if is_image_query else 'text'})")
                        print(f"  Query 2: {query2} ({'image' if is_image_query2 else 'text'})")
                        print(f"  Weights: {weights[0]:.1f} / {weights[1]:.1f}")
                        if negative_queries:
                            print(f"  Negatives ({len(negative_queries)}): {', '.join(negative_queries)}")
                        elif negative_query:
                            print(f"  Negative: {negative_query} ({'image' if negative_is_image else 'text'})")
                        print(f"  Number of results: {current_k}")
                    
                    results = db.search(query, k=current_k, is_image_path=is_image_query,
                                      query2=query2, is_image_path2=is_image_query2,
                                      weights=weights,
                                      negative_query=negative_query, negative_is_image=negative_is_image,
                                      negative_weight=negative_weight,
                                      negative_queries=negative_queries, negative_is_images=negative_is_images,
                                      negative_weights=negative_weights,
                                      filter_folders=filter_folders if filter_folders else None,
                                      profile=profile_enabled,
                                      show_duplicates=show_duplicates)
                    
                    if results:
                        print(f"\nFound {len(results)} results:")
                        for i, (file_path, similarity) in enumerate(results, 1):
                            print(f"  {i:2d}. {similarity:.4f}: {file_path}")
                        
                        # Always save results
                        output_file = generate_output_filename(query, is_image_query)
                        # Build query string for display (include query2 and negative if present)
                        display_query = query
                        if query2:
                            display_query += f" + {query2}"
                        if negative_queries:
                            display_query += " - " + " - ".join(negative_queries)
                        elif negative_query:
                            display_query += f" - {negative_query}"
                        db.generate_html_gallery(results, output_file, query=display_query)
                        print(f"\nResults saved to {output_file}")
                    else:
                        print("No results found.")
                    
                    # If not interactive, exit after processing the query
                    if not is_interactive:
                        break
                    
                    print()  # Blank line between queries
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Ending session.")
                    break
                except EOFError:
                    # If not interactive, exit silently (stdin not available)
                    if is_interactive:
                        print("\nEnding session. Goodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        else:
            # Single query mode
            if not args.query:
                print("Error: Query required (or use --interactive for session mode)")
                search_parser.print_help()
                return
            
            # Check for combined query
            if args.query2:
                print(f"Combined search:")
                print(f"  Query 1: {args.query} ({'image' if args.image else 'text'})")
                print(f"  Query 2: {args.query2} ({'image' if args.image2 else 'text'})")
                print(f"  Weights: {args.weights[0]:.1f} / {args.weights[1]:.1f}")
            
            # Check for negative query
            if args.negative:
                print(f"  Negative: {args.negative} ({'image' if args.negative_image else 'text'})")
            
            results = db.search(args.query, k=args.k, is_image_path=args.image,
                              query2=args.query2, is_image_path2=args.image2,
                              weights=tuple(args.weights),
                              negative_query=args.negative, negative_is_image=args.negative_image,
                              negative_weight=args.negative_weight,
                              filter_folders=args.folder if args.folder else None,
                              profile=args.profile,
                              show_duplicates=args.show_duplicates)
            
            if results:
                print(f"\nFound {len(results)} results:")
                for file_path, similarity in results:
                    print(f"  {similarity:.4f}: {file_path}")
                
                # Generate output filename from query, or use provided one
                if args.output == 'results.html':  # Default value
                    # For combined queries, create a combined filename
                    if args.query2:
                        query_name = f"{Path(args.query).stem if args.image else args.query[:50]}_and_{Path(args.query2).stem if args.image2 else args.query2[:50]}"
                        query_name = re.sub(r'[<>:"/\\|?*]', '_', query_name)
                        query_name = query_name.replace(' ', '_')[:100]
                        results_dir = Path(DEFAULT_RESULTS_DIR)
                        results_dir.mkdir(parents=True, exist_ok=True)
                        output_file = results_dir / f"{query_name}.html"
                        counter = 1
                        while output_file.exists():
                            counter += 1
                            output_file = results_dir / f"{query_name}_{counter}.html"
                        output_file = str(output_file)
                    else:
                        output_file = generate_output_filename(args.query, args.image)
                else:
                    output_file = args.output
                
                # Build query string for display (include query2 and negative if present)
                display_query = args.query
                if args.query2:
                    display_query += f" + {args.query2}"
                if args.negative:
                    display_query += f" - {args.negative}"
                
                db.generate_html_gallery(results, output_file, query=display_query)
                print(f"\nResults saved to {output_file}")
            else:
                print("No results found.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    print("Script starting...", flush=True)
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
