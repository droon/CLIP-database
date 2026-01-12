#!/usr/bin/env python3
"""
Searchable Image Database using SigLIP 2 and SQLite-vec

This script provides two modes:
1. Scan mode: Process images from a directory and store embeddings
2. Search mode: Search for similar images using text or image queries

NOTE: This code was generated with AI assistance.
"""

# ============================================================================
# Configuration Variables - Modify these paths as needed
# ============================================================================
# Default database path (SQLite database file)
DEFAULT_DB_PATH = "image_database.db"

# Default model cache directory (where HuggingFace models are stored)
DEFAULT_MODEL_CACHE_DIR = "models"

# Default results directory (where HTML search results are saved)
DEFAULT_RESULTS_DIR = "results"

# Default thumbnails directory (if thumbnails are generated)
DEFAULT_THUMBNAILS_DIR = "thumbnails"
# ============================================================================

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import numpy as np
import sqlite_vec


class ImageDatabase:
    """Manages the image database with SigLIP 2 embeddings and SQLite-vec."""
    
    def __init__(self, db_path: str = None, model_cache_dir: str = None):
        # Default paths
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        if model_cache_dir is None:
            model_cache_dir = DEFAULT_MODEL_CACHE_DIR
        
        self.db_path = db_path
        self.model_cache_dir = model_cache_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Initialize model
        print(f"Loading SigLIP 2 model on {self.device} with {self.dtype}...")
        
        # Try to load from local directory first, then fall back to HuggingFace
        local_model_path = os.path.join(model_cache_dir, "google--siglip2-so400m-patch14-224") if model_cache_dir else None
        model_name = "google/siglip2-so400m-patch14-224"
        
        if local_model_path and os.path.exists(local_model_path):
            print(f"Loading model from local directory: {local_model_path}")
            # Try explicit Siglip classes first to avoid Auto class confusion
            try:
                self.processor = SiglipProcessor.from_pretrained(local_model_path)
                self.model = SiglipModel.from_pretrained(local_model_path).to(self.device).to(self.dtype)
            except Exception as e:
                print(f"Error with explicit classes, trying Auto: {e}")
                self.processor = AutoProcessor.from_pretrained(local_model_path)
                self.model = AutoModel.from_pretrained(local_model_path).to(self.device).to(self.dtype)
        else:
            # Use default cache if model_cache_dir is None, otherwise use custom
            cache_kwargs = {}
            if model_cache_dir:
                cache_kwargs['cache_dir'] = model_cache_dir
                print(f"Model cache directory: {model_cache_dir}")
            
            # Try to load from HuggingFace
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, **cache_kwargs)
                self.model = AutoModel.from_pretrained(model_name, **cache_kwargs).to(self.device).to(self.dtype)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.model.eval()
        
        # Expected embedding dimension for SO400M
        self.embedding_dim = 1152
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with sqlite-vec extension."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        
        # Enable WAL mode for better concurrent access (reads can happen during writes)
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Enable extension loading (required for sqlite-vec)
        conn.enable_load_extension(True)
        
        # Load sqlite-vec extension
        try:
            sqlite_vec.load(conn)
            print("Loaded sqlite-vec extension")
        except Exception as e:
            print(f"ERROR: Could not load sqlite-vec extension: {e}")
            print("Please ensure sqlite-vec is installed: pip install sqlite-vec")
            sys.exit(1)
        
        cursor = conn.cursor()
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                last_modified REAL NOT NULL,
                file_hash TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create vec0 virtual table for embeddings
        # Only create if it doesn't exist (don't drop existing data!)
        try:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec0 USING vec0(
                    embedding float[{self.embedding_dim}]
                )
            """)
        except sqlite3.OperationalError as e:
            if "no such module: vec0" in str(e).lower() or "no such module" in str(e).lower():
                print("ERROR: sqlite-vec extension is not available!")
                print("Please install sqlite-vec and ensure it's accessible.")
                sys.exit(1)
            # If table already exists, that's fine
            if "already exists" not in str(e).lower():
                raise
        
        # Create mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                rowid INTEGER PRIMARY KEY,
                image_id INTEGER,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding from an image file."""
        try:
            image = Image.open(image_path).convert("RGB")
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
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _get_image_embeddings_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple images in a batch (much faster)."""
        images = []
        valid_paths = []
        
        # Load all images
        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
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
    
    def _is_image_processed(self, cursor, file_path: str, last_modified: float) -> bool:
        """Check if image is already processed with matching timestamp."""
        cursor.execute("""
            SELECT id FROM images 
            WHERE file_path = ? AND last_modified = ?
        """, (file_path, last_modified))
        return cursor.fetchone() is not None
    
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
            
            # Check if image exists AND has an embedding (both must be true)
            cursor.execute(f"""
                SELECT i.file_path 
                FROM images i
                JOIN image_embeddings ie ON i.id = ie.image_id
                WHERE (i.file_path, i.last_modified) IN (VALUES {placeholders})
            """, values)
            
            processed.update(row[0] for row in cursor.fetchall())
        
        return processed
    
    def scan_directory(self, root_dir: str, batch_size: int = 75, inference_batch_size: int = 16, profile: bool = False, limit: int = None):
        """
        Scan directory and process images.
        
        Args:
            root_dir: Root directory to scan
            batch_size: Number of images to process before committing to DB (default: 75)
            inference_batch_size: Number of images to process in parallel for model inference (default: 16)
            profile: If True, print timing information for each step
            limit: Limit number of images to process (for testing, None = no limit)
        """
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"Error: Directory {root_dir} does not exist")
            return
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        
        # First pass: count total files
        print("Counting image files...")
        # Use set to deduplicate (Windows is case-insensitive, so *.jpg and *.JPG find same files)
        image_files_set = set()
        for ext in image_extensions:
            # Search for both lowercase and uppercase, but deduplicate by absolute path
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for file_path in root_path.rglob(pattern):
                    image_files_set.add(file_path.absolute())  # Use absolute path for deduplication
        
        image_files = [Path(p) for p in image_files_set]  # Convert back to list of Path objects
        total_files = len(image_files)
        print(f"Found {total_files} unique image files")
        
        if total_files == 0:
            print("No image files found!")
            return
        
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
            # Pre-filter: batch check which images are already processed
            print("Checking which images are already processed...")
            to_process = []
            file_metadata = []
            
            check_start = time.time()
            for image_path in image_files:
                file_path = str(image_path.absolute())
                last_modified = os.path.getmtime(file_path)
                file_metadata.append((file_path, last_modified))
            
            # Batch check database (much faster than individual queries)
            processed_files = self._batch_check_processed(cursor, file_metadata)
            for file_path, last_modified in file_metadata:
                if file_path not in processed_files:
                    to_process.append((file_path, last_modified))
                else:
                    skipped += 1
            
            timers['check_db'] += time.time() - check_start
            timer_counts['check_db'] = 1  # One batch query instead of many
            
            # Apply limit if specified (for testing)
            if limit is not None:
                to_process = to_process[:limit]
                print(f"Limited to {limit} images for testing")
            
            print(f"Need to process {len(to_process)} images ({skipped} already done)")
            
            with tqdm(total=total_files, desc="Processing images") as pbar:
                pbar.update(skipped)  # Update for already processed
                
                # Process in inference batches (reuse variables initialized above)
                
                for file_path, last_modified in to_process:
                    inference_batch.append(file_path)
                    inference_metadata.append((file_path, last_modified))
                    
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
                            # Retry logic for database locks
                            max_retries = 5
                            for attempt in range(max_retries):
                                try:
                                    self._commit_batch(cursor, db_batch)
                                    conn.commit()
                                    break
                                except sqlite3.OperationalError as e:
                                    if "locked" in str(e).lower() and attempt < max_retries - 1:
                                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                        continue
                                    else:
                                        raise
                            timers['db_write'] += time.time() - db_start
                            timer_counts['db_write'] += len(db_batch)
                            processed += len(db_batch)
                            db_batch = []
                        
                        pbar.update(len(inference_batch))
                        inference_batch = []
                        inference_metadata = []
                
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
                    # Retry logic for database locks
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            self._commit_batch(cursor, db_batch)
                            conn.commit()
                            break
                        except sqlite3.OperationalError as e:
                            if "locked" in str(e).lower() and attempt < max_retries - 1:
                                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                continue
                            else:
                                raise
                    timers['db_write'] += time.time() - db_start
                    timer_counts['db_write'] += len(db_batch)
                    processed += len(db_batch)
        
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
                # Retry logic for database locks
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        self._commit_batch(cursor, db_batch)
                        conn.commit()
                        processed += len(db_batch)
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower() and attempt < max_retries - 1:
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            raise
            print(f"Progress saved: {processed} processed, {skipped} skipped, {errors} errors")
            print("You can resume by running the same command - already processed images will be skipped.")
        
        finally:
            conn.close()
        
        print(f"\nScan complete!")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors: {errors}")
        
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
    
    def _commit_batch(self, cursor, batch: List[Tuple]):
        """Commit a batch of images and embeddings to the database."""
        for file_path, last_modified, file_hash, embedding in batch:
            try:
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
                
                # Check if embedding already exists for this image
                cursor.execute("""
                    SELECT rowid FROM image_embeddings WHERE image_id = ?
                """, (image_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing embedding
                    vec_rowid = existing[0]
                    cursor.execute("""
                        UPDATE vec0 SET embedding = ? WHERE rowid = ?
                    """, (json.dumps(embedding.tolist()), vec_rowid))
                else:
                    # Insert new embedding
                    cursor.execute("""
                        INSERT INTO vec0 (embedding) VALUES (?)
                    """, (json.dumps(embedding.tolist()),))
                    
                    vec_rowid = cursor.lastrowid
                    
                    # Link embedding to image
                    cursor.execute("""
                        INSERT INTO image_embeddings (rowid, image_id)
                        VALUES (?, ?)
                    """, (vec_rowid, image_id))
            
            except sqlite3.IntegrityError:
                # Skip duplicates
                continue
            except Exception as e:
                print(f"Error committing {file_path}: {e}")
                continue
    
    def search(self, query: str, k: int = 10, is_image_path: bool = False, 
               query2: str = None, is_image_path2: bool = False, 
               weights: Tuple[float, float] = (0.5, 0.5)) -> List[Tuple[str, float]]:
        """
        Search for similar images. Supports combined queries (two images, or image + text).
        
        Args:
            query: Text string or image file path (first query)
            k: Number of results to return
            is_image_path: If True, treat query as image path; otherwise as text
            query2: Optional second query (text or image path)
            is_image_path2: If True, treat query2 as image path; otherwise as text
            weights: Tuple of (weight1, weight2) for combining queries. Default (0.5, 0.5)
        
        Returns:
            List of (file_path, similarity_score) tuples
        """
        # Get first query embedding
        if is_image_path:
            if not os.path.exists(query):
                print(f"Error: Image file {query} does not exist")
                return []
            embedding1 = self._get_image_embedding(query)
            if embedding1 is None:
                return []
        else:
            embedding1 = self._get_text_embedding(query)
        
        # If second query provided, combine embeddings
        if query2 is not None:
            # Get second query embedding
            if is_image_path2:
                if not os.path.exists(query2):
                    print(f"Error: Image file {query2} does not exist")
                    return []
                embedding2 = self._get_image_embedding(query2)
                if embedding2 is None:
                    return []
            else:
                embedding2 = self._get_text_embedding(query2)
            
            # Normalize weights
            total_weight = weights[0] + weights[1]
            if total_weight == 0:
                weights = (0.5, 0.5)
                total_weight = 1.0
            w1, w2 = weights[0] / total_weight, weights[1] / total_weight
            
            # Combine embeddings with weighted average
            embedding = w1 * embedding1 + w2 * embedding2
            
            # L2 normalize the combined embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                print("Warning: Combined embedding has zero norm, using first query only")
                embedding = embedding1
        else:
            embedding = embedding1
        
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable WAL mode for better concurrent access (allows reads during writes)
        conn.execute("PRAGMA journal_mode=WAL")
        # Ensure extension is loaded for this connection
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
            print("sqlite-vec extension loaded in search")
        except Exception as e:
            print(f"Warning: Could not load sqlite-vec in search: {e}")
        
        cursor = conn.cursor()
        
        # Verify vec0 is accessible
        try:
            cursor.execute("SELECT COUNT(*) FROM vec0")
            vec0_count = cursor.fetchone()[0]
            print(f"Vec0 table accessible, has {vec0_count} rows")
        except sqlite3.OperationalError as e:
            print(f"Error: vec0 table not accessible: {e}")
            conn.close()
            return []
        
        # Use sqlite-vec's built-in distance functions for efficient KNN search
        try:
            # Serialize the query embedding for sqlite-vec
            query_vec = sqlite_vec.serialize_float32(embedding)
            
            # Use vec_distance_cosine for cosine similarity (lower is more similar)
            # Note: vec_distance_cosine returns distance (0 = identical, 2 = opposite)
            # We convert to similarity: similarity = 1 - (distance / 2)
            cursor.execute("""
                SELECT 
                    i.file_path,
                    vec_distance_cosine(vec0.embedding, ?) as distance
                FROM vec0
                JOIN image_embeddings ie ON vec0.rowid = ie.rowid
                JOIN images i ON ie.image_id = i.id
                ORDER BY distance ASC
                LIMIT ?
            """, (query_vec, k))
            
            all_results = cursor.fetchall()
            if not all_results:
                print("No embeddings found in database (query returned 0 rows)")
                conn.close()
                return []
            
            # Convert distance to similarity score
            # vec_distance_cosine returns: distance = 1 - similarity
            # Therefore: similarity = 1 - distance
            results = []
            for file_path, distance in all_results:
                similarity = 1.0 - distance
                results.append((file_path, float(similarity)))
            
            if not results:
                print("No valid similarities calculated")
                conn.close()
                return []
        
        except Exception as e:
            print(f"Error during search: {e}")
            conn.close()
            return []
        
        conn.close()
        return results
    
    def generate_html_gallery(self, results: List[Tuple[str, float]], output_file: str = "results.html"):
        """Generate HTML gallery with original images (no thumbnails)."""
        
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
            
            # Convert file path to file:// URL for image display
            file_display_url = file_path.replace('\\', '/')
            if not file_display_url.startswith('/'):
                file_display_url = '/' + file_display_url
            file_display_url = f"file://{file_display_url}"
            
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
    results_dir.mkdir(exist_ok=True)
    
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
    parser = argparse.ArgumentParser(description="Searchable Image Database using SigLIP 2")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    # Scan mode
    scan_parser = subparsers.add_parser('scan', help='Scan directory and process images')
    scan_parser.add_argument('directory', help='Root directory to scan')
    scan_parser.add_argument('--db', default=DEFAULT_DB_PATH, help='Database path')
    scan_parser.add_argument('--batch-size', type=int, default=75, help='Batch size for DB commits')
    scan_parser.add_argument('--inference-batch-size', type=int, default=16, help='Batch size for model inference (higher = faster but more VRAM)')
    scan_parser.add_argument('--profile', action='store_true', help='Show performance profiling information')
    scan_parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process (for testing)')
    scan_parser.add_argument('--model-cache', default=DEFAULT_MODEL_CACHE_DIR, help='Model cache directory')
    
    # Search mode
    search_parser = subparsers.add_parser('search', help='Search for similar images')
    search_parser.add_argument('query', nargs='?', help='Text query or image file path (optional if using --interactive)')
    search_parser.add_argument('-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--image', action='store_true', help='Treat query as image file path')
    search_parser.add_argument('--query2', help='Second query for combined search (text or image path)')
    search_parser.add_argument('--image2', action='store_true', help='Treat query2 as image file path')
    search_parser.add_argument('--weights', nargs=2, type=float, default=[0.5, 0.5], metavar=('W1', 'W2'), help='Weights for combining queries (default: 0.5 0.5)')
    search_parser.add_argument('--db', default=DEFAULT_DB_PATH, help='Database path')
    search_parser.add_argument('--model-cache', default=DEFAULT_MODEL_CACHE_DIR, help='Model cache directory')
    search_parser.add_argument('--output', default='results.html', help='Output HTML file')
    search_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive session mode: load model once and run multiple queries')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        # Use None for model_cache to use default location if not specified
        model_cache = args.model_cache if args.model_cache else None
        db = ImageDatabase(args.db, model_cache)
        db.scan_directory(
            args.directory, 
            batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size,
            profile=args.profile,
            limit=args.limit
        )
    
    elif args.mode == 'search':
        # Use None for model_cache to use default location if not specified
        model_cache = args.model_cache if args.model_cache else None
        db = ImageDatabase(args.db, model_cache)
        
        if args.interactive:
            # Interactive session mode
            print("\n" + "="*60)
            print("Interactive Search Session")
            print("="*60)
            print("Model loaded and ready! Enter queries below.")
            print("Commands:")
            print("  - Enter a text query to search")
            print("  - Type 'image:<path>' to search by image")
            print("  - Type 'image:<path1> + image:<path2>' for combined image search")
            print("  - Type 'image:<path> + <text>' or '<text> + image:<path>' for image+text search")
            print("  - Type 'k:<number>' to change number of results (default: 10)")
            print("  - Type 'quit' or 'exit' to end session")
            print("="*60 + "\n")
            
            current_k = args.k
            is_image_query = args.image
            query2 = args.query2
            is_image_query2 = args.image2
            weights = tuple(args.weights)
            
            while True:
                try:
                    if args.query:
                        # Use provided query on first iteration
                        query = args.query
                        args.query = None  # Clear for next iterations
                    else:
                        query = input("Query> ").strip()
                    
                    if not query:
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
                    
                    # Parse combined queries (format: "query1 + query2")
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
                        print(f"Number of results: {current_k}")
                    
                    results = db.search(query, k=current_k, is_image_path=is_image_query,
                                      query2=query2, is_image_path2=is_image_query2,
                                      weights=weights)
                    
                    if results:
                        print(f"\nFound {len(results)} results:")
                        for i, (file_path, similarity) in enumerate(results, 1):
                            print(f"  {i:2d}. {similarity:.4f}: {file_path}")
                        
                        # Always save results
                        output_file = generate_output_filename(query, is_image_query)
                        db.generate_html_gallery(results, output_file)
                        print(f"\nResults saved to {output_file}")
                    else:
                        print("No results found.")
                    
                    print()  # Blank line between queries
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Ending session.")
                    break
                except EOFError:
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
            
            results = db.search(args.query, k=args.k, is_image_path=args.image,
                              query2=args.query2, is_image_path2=args.image2,
                              weights=tuple(args.weights))
            
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
                        results_dir.mkdir(exist_ok=True)
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
                
                db.generate_html_gallery(results, output_file)
                print(f"\nResults saved to {output_file}")
            else:
                print("No results found.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
