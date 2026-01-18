# Image Database Functionality Reference

## Configuration
- `load_config()` - Load configuration from config.json with fallback to defaults
- `resolve_path()` - Resolve config paths (absolute vs relative to base directory)

## ImageDatabase Class

### Initialization
- `__init__()` - Initialize database, load SigLIP 2 model, setup device (CUDA/CPU), create directories

### Database Management
- `_init_database()` - Initialize SQLite database with sqlite-vec extension, create tables, enable WAL mode
- `_commit_with_retry()` - Commit batch to database with retry logic for handling database locks
- `_commit_batch()` - Commit batch of images and embeddings to database (full and/or binary)

### File Utilities
- `_get_file_hash()` - Calculate SHA256 hash of file
- `_safe_print_path()` - Safely print file paths with Unicode character handling

### Embedding Generation
- `_get_image_embedding()` - Extract embedding from single image file
- `_get_image_embeddings_batch()` - Extract embeddings from multiple images in batch
- `_get_text_embedding()` - Extract embedding from text query (lowercase, padding, prompt template)
- `_apply_negative_embedding()` - Apply negative embedding subtraction and re-normalize

### Directory Scanning
- `_sample_folder_sequences()` - Sample sequences in folder (keep every 100th frame if >100 files)
- `_batch_check_processed()` - Batch check which images are already processed
- `scan_directory()` - Scan directory tree, process images, generate embeddings, store in database

### Search
- `search()` - Search for similar images using text/image queries with combined queries, negative prompts, folder filtering
- `generate_html_gallery()` - Generate HTML gallery with search results, images, similarity scores, file links

## Standalone Functions
- `generate_output_filename()` - Generate safe filename from query with auto-incrementing
- `main()` - Main CLI entry point with argument parsing for scan and search modes

## CLI Modes

### Scan Mode
- Process images from directory and generate embeddings
- Batch processing, exclusions, limits, profiling support
- Binary-only mode option

### Search Mode
- Text or image query search
- Combined queries (text+text, image+image, text+image)
- Negative prompts (exclude concepts)
- Folder filtering
- Interactive session mode
- HTML gallery generation
