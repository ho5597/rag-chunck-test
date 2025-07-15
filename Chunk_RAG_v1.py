"""
title: Llama Index Multi-Chunk RAG Pipeline
author: your-name
date: 2025-07-15
version: 1.0
license: MIT
description: A RAG pipeline that creates and queries vector indices with 3 different chunk sizes.
requirements: llama-index, llama-index-embeddings-openai
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.indexes = {}  # {chunk_size: VectorStoreIndex}
        self.chunk_sizes = [256, 512, 1024]

    async def on_startup(self):
        import os
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.openai import OpenAIEmbedding

        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        # Load all documents from ./data
        documents = SimpleDirectoryReader("./data").load_data()

        # Initialize 3 indices with different chunk sizes
        self.indexes = {}
        for chunk_size in self.chunk_sizes:
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes, embed_model=OpenAIEmbedding())
            self.indexes[chunk_size] = index

    async def on_shutdown(self):
        self.indexes = {}

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Embed the query
        embed_model = OpenAIEmbedding()
        query_vector = embed_model.get_query_embedding(user_message)

        best_score = float('-inf')
        best_engine = None

        for chunk_size, index in self.indexes.items():
            retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
            nodes = retriever.retrieve(user_message)

            if nodes:
                score = getattr(nodes[0], 'score', 0)
                if score > best_score:
                    best_score = score
                    best_engine = index.as_query_engine(streaming=False)

        if best_engine:
            response = best_engine.query(user_message)
            return response.response + "\n\nHello World!"
        else:
            return "No relevant document found in any chunk size index.\n\nHello World!"
