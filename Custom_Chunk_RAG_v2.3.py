"""
title: Llama Index Multi-Chunk Pipeline
author: open-webui
date: 2024-07-15
version: 2.3
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using Llama Index, supporting multiple chunk sizes with dynamic overlap.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
# ... other imports remain the same ...

class Pipeline:
    def __init__(self):
        self.documents = None
        self.vector_indices = {}
        self.query_engines = {}
        self.chunk_sizes = [256, 512, 1024] 

    async def on_startup(self):
        os.environ["OPENAI_API_KEY"] = "your-api-key-here" 
        knowledge_base_path = "./workspace/knowledge/" 
        
        # Verify the path exists before trying to read from it
        if not os.path.isdir(knowledge_base_path):
            print(f"Error: Knowledge base directory not found at {knowledge_base_path}. Please verify the path.")
            # Depending on the error you get, you might need to adjust knowledge_base_path
            # Or you might need to raise an exception to stop the pipeline from starting.
            return # Exit if the path is invalid

        print(f"Loading documents from Open WebUI's Knowledge base: {knowledge_base_path}...")
        self.documents = SimpleDirectoryReader(knowledge_base_path).load_data()
        print(f"Loaded {len(self.documents)} documents.")

        for chunk_size in self.chunk_sizes:
            chunk_overlap = int(chunk_size * 0.15) 
            print(f"Processing documents with chunk size: {chunk_size}, overlap: {chunk_overlap}...")
            
            node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = node_parser.get_nodes_from_documents(self.documents)

            index_name = f"index_chunk_{chunk_size}"
            self.vector_indices[index_name] = VectorStoreIndex(nodes)
            print(f"Created index for chunk size {chunk_size}.")

            self.query_engines[index_name] = self.vector_indices[index_name].as_query_engine(streaming=True)
            print(f"Created query engine for chunk size {chunk_size}.")

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"User message: {user_message}")
        print(f"Model ID: {model_id}")

        best_response = None
        for chunk_size in self.chunk_sizes:
            index_name = f"index_chunk_{chunk_size}"
            print(f"Attempting to query with chunk size: {chunk_size}")
            try:
                query_engine = self.query_engines[index_name]
                response = query_engine.query(user_message)
                
                if response and response.response_gen:
                    print(f"Found response using chunk size: {chunk_size}")
                    best_response = response.response_gen
                    break 
            except Exception as e:
                print(f"Error querying with chunk size {chunk_size}: {e}")
                continue

        if best_response:
            return best_response
        else:
            return "I couldn't find relevant information from the knowledge base across any chunk sizes."
