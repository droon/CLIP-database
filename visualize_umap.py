#!/usr/bin/env python3
"""
Generate a UMAP 3D visualization of image embeddings with clustering and image preview

NOTE: This code was generated with AI assistance.
"""

# ============================================================================
# Configuration Variables - Modify these paths as needed
# ============================================================================
# Database path (SQLite database file)
DB_PATH = "image_database.db"

# Output file for the 3D visualization
OUTPUT_HTML_FILE = "umap_3d_visualization.html"

# Cache file for UMAP projections (to avoid recomputing)
UMAP_CACHE_FILE = "umap_projections_cache.pkl"

# Image metadata JSON file
IMAGE_METADATA_FILE = "umap_image_metadata.json"
# ============================================================================

import sqlite3
import sqlite_vec
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import os

print("Loading embeddings from database...")
conn = sqlite3.connect(DB_PATH, timeout=30.0)
conn.execute("PRAGMA journal_mode=WAL")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
cursor = conn.cursor()

# Get all embeddings with their file paths
cursor.execute("""
    SELECT 
        i.file_path,
        vec0.embedding
    FROM vec0
    JOIN image_embeddings ie ON vec0.rowid = ie.rowid
    JOIN images i ON ie.image_id = i.id
""")

print("Fetching all embeddings...")
all_results = cursor.fetchall()
conn.close()

if not all_results:
    print("No embeddings found in database!")
    exit(1)

print(f"Found {len(all_results):,} embeddings")

# Extract vectors and paths
print("Converting embeddings to numpy arrays...")
all_vectors = []
all_image_paths = []

for file_path, emb_data in tqdm(all_results, desc="Processing embeddings"):
    try:
        # Convert BLOB to numpy array
        emb = np.frombuffer(emb_data, dtype=np.float32)
        if emb.shape[0] == 1152:  # Verify dimension
            all_vectors.append(emb)
            all_image_paths.append(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

if not all_vectors:
    print("No valid embeddings found!")
    exit(1)

all_vectors = np.array(all_vectors)
print(f"Embedding matrix shape: {all_vectors.shape}")

# Fit UMAP (Takes ~5-10 mins for 100k points on a good CPU)
print("\nFitting UMAP reducer (this may take a while)...")
reducer = umap.UMAP(
    n_neighbors=15,      # Balance between local and global structure
    min_dist=0.1,        # How tightly points pack together
    n_components=3,      # For a 3D plot
    metric='cosine',     # Best for SigLIP/CLIP vectors
    verbose=True
)

# Check if we have saved UMAP projections
use_cache = False

if os.path.exists(UMAP_CACHE_FILE):
    print(f"\nLoading cached UMAP projections from {UMAP_CACHE_FILE}...")
    try:
        with open(UMAP_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            if (cache_data.get('vectors_hash') == hash(all_vectors.tobytes()) and 
                len(cache_data.get('projections', [])) == len(all_vectors)):
                projections = cache_data['projections']
                print("Using cached UMAP projections (no recalculation needed!)")
                use_cache = True
            else:
                print("Cache outdated, will recalculate UMAP...")
    except Exception as e:
        print(f"Error loading cache: {e}, will recalculate UMAP...")

if not use_cache:
    print("\nFitting UMAP reducer (this may take a while)...")
    projections = reducer.fit_transform(all_vectors)
    print(f"UMAP projections shape: {projections.shape}")
    # Save for future use
    print(f"Saving UMAP projections to {UMAP_CACHE_FILE} for future use...")
    with open(UMAP_CACHE_FILE, 'wb') as f:
        pickle.dump({
            'projections': projections,
            'vectors_hash': hash(all_vectors.tobytes())
        }, f)

# Cluster the embeddings for coloring (fast, separate from UMAP)
# Note: Clustering happens in 1152D space, but visualization is in 3D UMAP space.
# This means clusters may appear intermingled because:
# 1. UMAP preserves local structure but not global cluster boundaries
# 2. 3D projection loses information from 1152 dimensions
# 3. Points close in 1152D might be far in 3D, and vice versa
# This is normal and expected behavior!
print("\nClustering embeddings for color coding...")
n_clusters = min(20, len(all_vectors) // 100)  # Adaptive number of clusters
if n_clusters < 2:
    n_clusters = 2

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_vectors)
print(f"Created {n_clusters} clusters")
print("Note: Clusters are computed in 1152D embedding space, but visualized in 3D UMAP space.")
print("      Some intermingling is expected - clusters represent semantic similarity in high-D space.")

# Create DataFrame
print("\nCreating 3D visualization...")
df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
df['path'] = all_image_paths
df['cluster'] = cluster_labels
df['filename'] = df['path'].apply(lambda p: p.split('\\')[-1] if '\\' in p else p.split('/')[-1])
# Convert file paths to file:// URLs for direct image display
df['image_url'] = df['path'].apply(lambda p: f"file:///{p.replace(chr(92), '/')}")

# Create 3D scatter plot with cluster colors
fig = px.scatter_3d(
    df, 
    x='x', 
    y='y', 
    z='z',
    color='cluster',
    color_discrete_sequence=px.colors.qualitative.Set3,
    hover_name='filename',
    hover_data={'path': True, 'cluster': True, 'filename': False},
    title=f"UMAP Photo Universe ({len(df):,} images, {n_clusters} clusters)",
    opacity=0.6,
    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3', 'cluster': 'Cluster'},
    custom_data=['path', 'image_url', 'filename']
)

# Add JavaScript for sidebar image display
fig.update_layout(
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3',
    ),
    width=1400,
    height=800
)

# Save image metadata to separate JSON file (much smaller)
import json
print(f"Saving image metadata to {IMAGE_METADATA_FILE}...")
image_metadata = df[['path', 'image_url', 'filename']].to_dict('records')
with open(IMAGE_METADATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(image_metadata, f)

# Use Plotly's efficient HTML writer (uses CDN, doesn't embed plotly.js)
print(f"Generating HTML with Plotly's efficient writer...")
fig.write_html(
    OUTPUT_HTML_FILE,
    include_plotlyjs='cdn',  # Use CDN instead of embedding (much smaller file)
    div_id='plot'
)

# Post-process: add sidebar and load metadata from external JSON
print("Adding custom sidebar functionality...")
with open(OUTPUT_HTML_FILE, 'r', encoding='utf-8') as f:
    html = f.read()

# Add CSS before </head>
css = """    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #plot-container { display: flex; height: 100vh; }
        #plot { flex: 1; }
        #sidebar { width: 350px; background: #f5f5f5; padding: 20px; overflow-y: auto; border-left: 1px solid #ddd; }
        #sidebar.hidden { display: none; }
        #sidebar img { max-width: 100%; max-height: 400px; width: auto; height: auto; border-radius: 8px; margin-bottom: 10px; object-fit: contain; }
        #sidebar .info { font-size: 12px; color: #666; word-break: break-all; }
        #sidebar h3 { margin-top: 0; }
        #sidebar .close { float: right; cursor: pointer; font-size: 20px; }
    </style>
"""
html = html.replace('</head>', css + '</head>')

# Wrap plot div in container and add sidebar before </body>
sidebar_html = """        <div id="sidebar" class="hidden">
            <span class="close" onclick="closeSidebar()">×</span>
            <h3>Image Preview</h3>
            <div id="image-container"></div>
            <div id="info-container" class="info"></div>
        </div>
    </div>
"""
# Simple approach: wrap body content in plot-container
# Find where the plot div starts (Plotly puts it right after <body>)
html = html.replace('<body>', '<body>\\n    <div id="plot-container">')
# Add sidebar before closing body
html = html.replace('</body>', sidebar_html + '</body>')

# Add JavaScript before </body>
# Use customdata that's already embedded in the plot (more reliable than external JSON)
js = """    <script>
        function setupClickHandler() {
            var plotDiv = document.getElementById('plot');
            if (plotDiv && plotDiv.data) {
                // Attach click handler after plot is ready
                plotDiv.on('plotly_click', function(data) {
                    if (data && data.points && data.points.length > 0) {
                        var point = data.points[0];
                        var customData = point.customdata;
                        
                        if (customData && customData.length >= 3) {
                            var path = customData[0];
                            var imageUrl = customData[1];
                            var filename = customData[2];
                            
                            var sidebar = document.getElementById('sidebar');
                            var imageContainer = document.getElementById('image-container');
                            var infoContainer = document.getElementById('info-container');
                            
                            if (sidebar && imageContainer && infoContainer) {
                                imageContainer.innerHTML = '<img src="' + imageUrl + '" alt="Preview" loading="lazy" onerror="this.src=\\'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2RkZCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+\\';">';
                                infoContainer.innerHTML = '<strong>File:</strong><br>' + filename + '<br><br><strong>Path:</strong><br>' + path;
                                sidebar.classList.remove('hidden');
                            }
                        }
                    }
                });
            } else {
                // Retry if plot not ready yet
                setTimeout(setupClickHandler, 100);
            }
        }
        
        // Wait for DOM and Plotly to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(setupClickHandler, 500);
            });
        } else {
            setTimeout(setupClickHandler, 500);
        }
        
        function closeSidebar() {
            var sidebar = document.getElementById('sidebar');
            if (sidebar) {
                sidebar.classList.add('hidden');
            }
        }
    </script>
"""
html = html.replace('</body>', js + '</body>')

with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n3D visualization saved to {OUTPUT_HTML_FILE}")
print("Open it in your browser to explore!")
print("Click on any point to see the image in the sidebar!")
