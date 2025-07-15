"""
title: Dynamic Chunk RAG
author: hyemin-oh
version: 1.2
license: MIT
description: Use Open WebUI knowledge with dynamic chunk size control in the user prompt and test with unicorn message.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage

class Pipeline:
    def __init__(self):
        pass

    async def on_startup(self):
        import os
        # Automatically read from server environment
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    async def on_shutdown(self):
        pass

    def extract_chunk_size(self, message: str) -> int:
        import re
        match = re.search(r"\[chunk:(\d+)]", message)
        return int(match.group(1)) if match else 512  # Default to 512 if not specified

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        import re, math
        from llama_index.core import VectorStoreIndex, ServiceContext
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document
        from llama_index.embeddings.openai import OpenAIEmbedding

        # 1. Extract chunk size from user prompt
        chunk_size = self.extract_chunk_size(user_message)
        query_text = re.sub(r"\[chunk:\d+]", "", user_message).strip()

        # 2. Get knowledge data passed from WebUI
        knowledge = body.get("knowledge", [])
        if not knowledge:
            return "[Error] No knowledge provided from WebUI. Please select a document in the workspace."

        # 3. Convert WebUI documents into LlamaIndex documents
        docs = [Document(text=item["content"], metadata=item.get("meta", {})) for item in knowledge]

        # 4. Build the custom index with given chunk size and 20% overlap
        chunk_overlap = max(1, math.floor(chunk_size * 0.2))
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        service_context = ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(),
            node_parser=splitter
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
        # 5. Query and modify the response
        query_engine = index.as_query_engine(streaming=False)  # ← must be False!
        response = query_engine.query(query_text)
        
        # 6. Add test string
        return response.response + "\n\n🦄 Unicorn does exist!"
