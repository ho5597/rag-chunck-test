"""
title: Dynamic Chunk RAG
author: hyemin-oh
version: 1.0
license: MIT
description: Use Open WebUI knowledge with dynamic chunk size control in the user prompt.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        import os
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key")

    async def on_shutdown(self):
        pass

    def extract_chunk_size(self, message: str) -> int:
        import re
        match = re.search(r"\[chunk:(\d+)]", message)
        return int(match.group(1)) if match else 512  # default chunk size

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        import re
        from llama_index.core import VectorStoreIndex, ServiceContext
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document
        from llama_index.embeddings.openai import OpenAIEmbedding

        # 1. Get chunk size from prompt
        chunk_size = self.extract_chunk_size(user_message)
        query_text = re.sub(r"\[chunk:\d+]", "", user_message).strip()

        # 2. Collect knowledge data from WebUI context
        knowledge = body.get("knowledge", [])
        if not knowledge:
            return "[Error] No knowledge provided from WebUI. Please select a document in the workspace."

        docs = [Document(text=item["content"], metadata=item.get("meta", {})) for item in knowledge]

        # 3. Build index with custom chunk size
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
        service_context = ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(),
            node_parser=splitter
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        # 4. Query
        query_engine = index.as_query_engine(streaming=True)
        response = query_engine.query(query_text)

        return response.response_gen
