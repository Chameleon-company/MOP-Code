"""
LM Studio integration for LlamaIndex.

Custom LLM and Embedding classes that bypass LlamaIndex's model name
validation and call LM Studio's OpenAI-compatible API directly.
"""

from typing import Any, List, Optional, Sequence

import requests
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback

from core.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_LLM_MODEL,
    LM_STUDIO_EMBED_MODEL,
    REPORT_LLM_TIMEOUT,
)


class LMStudioEmbedding(BaseEmbedding):
    """Embedding class that calls LM Studio's endpoint directly."""

    _api_url: str = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self, api_base: str = LM_STUDIO_BASE_URL, model: str = LM_STUDIO_EMBED_MODEL
    ):
        super().__init__(model_name=model, embed_batch_size=10)
        self._api_url = f"{api_base}/embeddings"
        self._model = model

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            self._api_url,
            json={"model": self._model, "input": texts},
            timeout=REPORT_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._call_api([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._call_api([text])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


class LMStudioLLM(CustomLLM):
    """LLM class that calls LM Studio's chat completions endpoint."""

    model_name: str = LM_STUDIO_LLM_MODEL
    api_base: str = LM_STUDIO_BASE_URL
    max_tokens: int = 2048
    temperature: float = 0.3
    context_window: int = 8192

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=REPORT_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=text)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        api_messages = []
        for msg in messages:
            api_messages.append(
                {
                    "role": (
                        msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    ),
                    "content": msg.content,
                }
            )

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model_name,
                "messages": api_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=REPORT_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return ChatResponse(
            message=ChatMessage(role="assistant", content=text),
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        # Non-streaming fallback
        return self.complete(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        # Non-streaming fallback
        return self.chat(messages, **kwargs)
