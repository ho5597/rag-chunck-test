"""
title: Multi-Chunk Knowledge RAG Pipeline
author: your-name
version: 2.1
license: MIT
description: A RAG pipeline that chunks uploaded knowledge files into 3 sizes (256, 512, 1024) with 15% overlap and selects the most relevant at query time.
requirements: llama-index, llama-index-embeddings-openai
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from server.chat.utils import context


class Pipeline:
    def __init__(self):
        self.indexes = {}  # {filename_chunkSize: VectorStoreIndex}
        self.chunk_sizes = [256, 512, 1024]

    async def on_startup(self):
        import os
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.embeddings.openai import OpenAIEmbedding

        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        # Fetch knowledge files uploaded via Workspace > Knowledge
        knowledge_files = await context.get_knowledge_files()
        self.indexes = {}

        for file in knowledge_files:
            content = await file.get_content()
            if not content:
                continue  # skip empty files

            doc = Document(text=content)

            for chunk_size in self.chunk_sizes:
                overlap = int(chunk_size * 0.15)
                parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                nodes = parser.get_nodes_from_documents([doc])

                index = VectorStoreIndex(nodes, embed_model=OpenAIEmbedding())
                key = f"{file.name}_chunk:{chunk_size}"
                self.indexes[key] = index

    async def on_shutdown(self):
        self.indexes = {}

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not self.indexes:
            return "No indexed knowledge available. Please upload documents in the Knowledge tab."

        embed_model = OpenAIEmbedding()
        query_vector = embed_model.get_query_embedding(user_message)

        best_score = float('-inf')
        best_engine = None
        best_key = None

        for key, index in self.indexes.items():
            retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
            nodes = retriever.retrieve(user_message)

            if nodes:
                score = getattr(nodes[0], 'score', 0)
                if score > best_score:
                    best_score = score
                    best_engine = index.as_query_engine(streaming=False)
                    best_key = key

        if best_engine:
            response = best_engine.query(user_message)
            return f"[From {best_key}]\n\n{response.response}\n\nHello World!"
        else:
            return "No relevant information found in any chunked index.\n\nHello World!"
