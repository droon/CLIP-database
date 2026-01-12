# CLIP Image Database

A searchable image database using SigLIP 2 (CLIP) embeddings and SQLite-vec for efficient similarity search.

**Note: This code was generated with AI assistance.**

## Features

- **Image Indexing**: Scan directories and extract CLIP embeddings from images
- **Text Search**: Search images using natural language queries
- **Image Search**: Find similar images using a reference image
- **Combined Search**: Combine text and image queries with weighted blending
- **Interactive Mode**: Load model once and run multiple queries
- **3D Visualization**: UMAP-based 3D visualization of image embeddings with clustering

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- See `requirements.txt` for Python dependencies

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd CLIP-database
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install sqlite-vec extension (if not already installed):
```bash
pip install sqlite-vec
```

## Configuration

Edit the configuration variables at the top of each Python file to customize paths:

### `image_database.py`
- `DEFAULT_DB_PATH`: Path to SQLite database file
- `DEFAULT_MODEL_CACHE_DIR`: Directory for HuggingFace model cache
- `DEFAULT_RESULTS_DIR`: Directory for HTML search results
- `DEFAULT_THUMBNAILS_DIR`: Directory for thumbnails (if used)

### `visualize_umap.py`
- `DB_PATH`: Path to SQLite database file
- `OUTPUT_HTML_FILE`: Output file for 3D visualization
- `UMAP_CACHE_FILE`: Cache file for UMAP projections
- `IMAGE_METADATA_FILE`: JSON file for image metadata

## Usage

### Indexing Images

Scan a directory and build the image database:

```bash
python image_database.py scan /path/to/images --db image_database.db --model-cache models
```

Options:
- `--batch-size`: Number of images to process before committing to DB (default: 75)
- `--inference-batch-size`: Batch size for model inference (default: 16, higher = faster but more VRAM)
- `--profile`: Show performance profiling information
- `--limit`: Limit number of images to process (for testing)

### Searching Images

#### Text Search
```bash
python image_database.py search "a red car" -k 20
```

#### Image Search
```bash
python image_database.py search /path/to/image.jpg --image -k 20
```

#### Combined Search
```bash
python image_database.py search "sunset" --query2 /path/to/image.jpg --weights 0.7 0.3 -k 20
```

#### Interactive Mode
```bash
python image_database.py search --interactive
```

In interactive mode:
- Enter text queries directly
- Use `image:/path/to/image.jpg` for image queries
- Combine queries with `+`: `image:/path/to/img.jpg + sunset`
- Change result count with `k:20`
- Type `quit` or `exit` to end session

### 3D Visualization

Generate a UMAP 3D visualization of all image embeddings:

```bash
python visualize_umap.py
```

This will:
1. Load embeddings from the database
2. Compute UMAP projections (cached for future runs)
3. Cluster embeddings for color coding
4. Generate an interactive HTML visualization

Open the generated HTML file in your browser and click on points to see image previews.

## Model

This project uses [SigLIP 2 SO400M](https://huggingface.co/google/siglip2-so400m-patch14-224) from Google, which provides:
- 1152-dimensional embeddings
- Strong text-image alignment
- Efficient inference

The model will be automatically downloaded from HuggingFace on first use (or use `--model-cache` to specify a custom cache directory).

## Database Schema

The SQLite database contains:
- `images` table: Image metadata (file path, last modified, hash)
- `vec0` virtual table: Vector embeddings (using sqlite-vec)
- `image_embeddings` table: Mapping between images and embeddings

## Performance Tips

- Use `--inference-batch-size` to optimize GPU memory usage
- Enable `--profile` to identify bottlenecks
- The database uses WAL mode for better concurrent access
- UMAP projections are cached to avoid recomputation

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]
