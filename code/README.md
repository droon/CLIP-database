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
- **HTML Gallery**: Beautiful search results with image previews and direct file access

![Search Results Example](browser.png)

**Note:** The HTML search results use `localexplorer:` protocol links for opening files and folders. To use these links, you'll need a browser extension like [Local Explorer](https://chrome.google.com/webstore/detail/local-explorer/llbiblehpbpeflfgjcdfcpcakjhddedi) for Chrome/Edge or similar extensions for other browsers. Without the extension, you can still view images and copy file paths manually.

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
cd code
pip install -r requirements.txt
```

3. Install sqlite-vec extension (if not already installed):
```bash
pip install sqlite-vec
```

4. Configure paths by creating `config.json` in the project root:
```bash
# Copy the example config file
cp code/config.json.example ../config.json
# Then edit config.json with your preferred paths
```

Or create `config.json` manually in the project root:
```json
{
  "database_path": "image_database.db",
  "model_cache_dir": "models",
  "results_dir": "results",
  "thumbnails_dir": "thumbnails",
  "umap_output_file": "umap_3d_visualization.html",
  "umap_cache_file": "umap_projections_cache.pkl",
  "umap_metadata_file": "umap_image_metadata.json"
}
```

All paths in `config.json` are relative to the project root directory. The code automatically reads from this configuration file. If `config.json` doesn't exist, it will use default values.

**Note:** `config.json` is not tracked by git (it's in `.gitignore`), so you can safely store your personal paths without exposing them publicly. The `config.json.example` file shows the expected format.

## Project Structure

```
CLIP-database/
├── code/                    # Public code (tracked by git)
│   ├── image_database.py   # Main database script
│   ├── visualize_umap.py   # UMAP visualization script
│   ├── requirements.txt    # Python dependencies
│   ├── config.json.example  # Example config file (committed to git)
│   └── README.md          # This file
├── config.json             # Your personal config (not tracked, create from example)
├── image_database.db        # SQLite database (created automatically)
├── models/                  # Model cache directory (created automatically)
├── results/                 # Search results HTML files (created automatically)
└── umap_3d_visualization.html  # UMAP visualization output
```

## Usage

### Indexing Images

Scan a directory and build the image database:

```bash
cd code
python image_database.py scan /path/to/images
```

The script will automatically use paths from `config.json` in the project root. You can override them with command-line arguments if needed:
```bash
python image_database.py scan /path/to/images --db ../image_database.db --model-cache ../models
```

Options:
- `--batch-size`: Number of images to process before committing to DB (default: 75)
- `--inference-batch-size`: Batch size for model inference (default: 16, higher = faster but more VRAM)
- `--profile`: Show performance profiling information
- `--limit`: Limit number of images to process (for testing)

### Searching Images

#### Text Search
```bash
cd code
python image_database.py search "a red car" -k 20
```

#### Image Search
```bash
cd code
python image_database.py search /path/to/image.jpg --image -k 20
```

#### Combined Search
```bash
cd code
python image_database.py search "sunset" --query2 /path/to/image.jpg --weights 0.7 0.3 -k 20
```

#### Negative Prompts (Exclusion)
Use negative prompts to exclude unwanted concepts from your search results. The system subtracts the negative embedding from your query, moving results away from the negative concept.

**Text negative prompt:**
```bash
cd code
python image_database.py search "beautiful landscape" --negative "people" -k 20
```

**Image negative prompt:**
```bash
cd code
python image_database.py search "sunset" --negative /path/to/unwanted_image.jpg --negative-image -k 20
```

**Adjust negative weight:**
```bash
cd code
python image_database.py search "nature" --negative "buildings" --negative-weight 0.7 -k 20
```

The `--negative-weight` parameter (default: 0.5) controls how strongly the negative prompt affects results. Higher values (0.7-1.0) exclude the negative concept more aggressively, while lower values (0.2-0.4) provide subtle filtering.

**Combined with other features:**
```bash
cd code
python image_database.py search "ocean" --query2 /path/to/reference.jpg --negative "boats" --weights 0.6 0.4 --negative-weight 0.6 -k 20
```

#### Interactive Mode
```bash
cd code
python image_database.py search --interactive
```

In interactive mode:
- Enter text queries directly
- Use `image:/path/to/image.jpg` for image queries
- Combine queries with `+`: `image:/path/to/img.jpg + sunset`
- Use negative prompts with ` - `: `beautiful landscape - people` or `sunset - image:/path/to/unwanted.jpg`
- Change result count with `k:20`
- Type `quit` or `exit` to end session

**Negative prompt examples in interactive mode:**
- `ocean - boats` (exclude boats from ocean images)
- `nature - buildings` (find nature without buildings)
- `portrait - image:/path/to/bad_example.jpg` (exclude similar style to bad example)
- `sunset + ocean - people` (combine positive queries and exclude people)

### 3D Visualization

Generate a UMAP 3D visualization of all image embeddings:

```bash
cd code
python visualize_umap.py
```

This will:
1. Load embeddings from the database
2. Compute UMAP projections (cached for future runs)
3. Cluster embeddings for color coding
4. Generate an interactive HTML visualization

Open the generated HTML file in your browser and click on points to see image previews.

**Note:** The HTML results include "Open Image" and "Open Folder" links that use the `localexplorer:` protocol. To use these links, install a browser extension like [Local Explorer](https://chrome.google.com/webstore/detail/local-explorer/llbiblehpbpeflfgjcdfcpcakjhddedi) for Chrome/Edge. Without the extension, images will still display, but the file/folder links won't work.

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

MIT License - see [LICENSE](LICENSE) file for details.

This project uses:
- [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-224) - Apache 2.0 License
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - MIT/Apache 2.0 dual license
- SQLite - Public Domain
