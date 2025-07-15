from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import SentenceSplitter
import os

# Existing path where Open WebUI stores the uploaded documents
path = "./workspace/knowledge"

# Load all documents
documents = SimpleDirectoryReader(path).load_data()

# Loop over multiple chunk sizes
chunk_sizes = [256, 512, 1024]
overlap = 0.15

for size in chunk_sizes:
    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=int(size * overlap))
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Optional: use index metadata or naming to separate each chunk size
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=f"./storage_{size}")
