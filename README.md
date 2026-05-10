# Image Semantic Search

A Python library for semantic search and retrieval of images using CLIP embeddings and vector databases.

## Features

- **CLIP-based Embeddings**: Uses OpenAI's CLIP model for generating semantic embeddings from images and text
- **Vector Search**: Powered by Qdrant vector database for fast similarity search
- **Flexible Input**: Support for image directories, individual images, and custom metadata
- **Multiple Query Types**: Search by text queries or find similar images
- **LangChain Integration**: Compatible with LangChain retriever interface
- **Batch Processing**: Efficient processing of large image datasets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-semantic-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from image_retriever import ImageRetriever

# Initialize retriever
retriever = ImageRetriever(collection_name="my_images")

# Add images from directory
retriever.add_image_directory("/path/to/your/images")

# Search for images
results = retriever.search("a cat playing with yarn", limit=5)

for result in results:
    print(f"Found: {result['filename']} (score: {result['score']:.3f})")
```

### Command Line Usage

Create sample dataset and search:
```bash
python main.py --create-dataset --query "cat playing"
```

Search existing images:
```bash
python main.py --dataset-dir /path/to/images --query "sunset over ocean"
```

Find similar images:
```bash
python main.py --dataset-dir /path/to/images --image-query /path/to/query.jpg
```

## API Reference

### ImageRetriever

#### Initialization
```python
retriever = ImageRetriever(
    collection_name="my_collection",  # Qdrant collection name
    qdrant_url=None,                  # Qdrant server URL (None for in-memory)
    qdrant_api_key=None,              # Qdrant API key
    device="auto",                    # Device: 'auto', 'cpu', 'cuda'
    model_name="ViT-B-32",            # CLIP model name
    pretrained="openai"               # CLIP pretrained weights
)
```

#### Methods

- `add_images(image_paths, metadata=None, batch_size=32)`: Add list of images
- `add_image_directory(directory, extensions=['.jpg', '.jpeg', '.png'], recursive=True, metadata_func=None)`: Add all images from directory
- `search(query, limit=10, score_threshold=None)`: Search by text query
- `search_by_image(image_path, limit=10, score_threshold=None)`: Find similar images
- `get_collection_info()`: Get collection statistics
- `delete_collection()`: Delete the collection
- `save_index(path)`: Save collection to disk (local Qdrant only)
- `load_index(path)`: Load collection from disk (local Qdrant only)

### LangChain Integration

```python
from image_retriever import ImageRetriever, ImageRetrieverLangChain

# Create retriever
base_retriever = ImageRetriever()
base_retriever.add_image_directory("images")

# Wrap for LangChain
lc_retriever = ImageRetrieverLangChain(base_retriever)

# Use with LangChain
results = lc_retriever.get_relevant_documents("search query")
```

## Configuration

### CLIP Models

The retriever supports different CLIP models:

- `ViT-B-32` (default): Balanced performance and speed
- `ViT-B-16`: Higher accuracy, slower
- `ViT-L-14`: Best accuracy, slowest
- `RN50`, `RN101`, `RN50x4`, `RN50x16`: ResNet variants

### Vector Database

- **In-memory**: Default, fast but not persistent
- **Qdrant Cloud**: Set `qdrant_url` and `qdrant_api_key`
- **Local Qdrant**: Run Qdrant server locally

## Dependencies

- torch: PyTorch for CLIP model
- open_clip_torch: OpenCLIP implementation
- qdrant-client: Vector database client
- pillow: Image processing
- numpy: Numerical operations
- langchain-core: LangChain integration (optional)

## Performance Tips

1. **Batch Processing**: Use larger batch sizes for better GPU utilization
2. **Image Preprocessing**: Ensure images are in RGB format
3. **Index Persistence**: Use Qdrant server for large datasets
4. **Memory Management**: Process large datasets in chunks

## Examples

See `main.py` for command-line examples and usage patterns.

## License

MIT License