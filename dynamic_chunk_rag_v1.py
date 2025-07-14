"""
title: Dynamic Chunk RAG with Proportional Overlap
author: hyemin-oh
version: 1.1
license: MIT
description: Use WebUI knowledge with dynamic chunk size.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage

class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        import os
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    async def on_shutdown(self):
        pass

    def extract_chunk_size(self, message: str) -> int:
        import re
        match = re.search(r"\[chunk:(\d+)]", message)
        return int(match.group(1)) if match else 512

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        import re, math
        from llama_index.core import VectorStoreIndex, ServiceContext
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document
        from llama_index.embeddings.openai import OpenAIEmbedding

        # 1. Extract chunk size
        chunk_size = self.extract_chunk_size(user_message)
        overlap = max(1, math.floor(chunk_size * 0.15))

        # 2. Clean query text
        query_text = re.sub(r"\[chunk:\d+]", "", user_message).strip()

        # 3. Load selected knowledge
        knowledge = body.get("knowledge", [])
        if not knowledge:
            return "[Error] No knowledge selected. Please pick document(s) in WebUI."

        docs = [Document(text=item["content"], metadata=item.get("meta", {})) 
                for item in knowledge]

        # 4. Build index with dynamic overlap
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        service_context = ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(),
            node_parser=splitter
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        # 5. Retrieve answer
        query_engine = index.as_query_engine(streaming=True)
        response = query_engine.query(query_text)

        return response.response_gen
