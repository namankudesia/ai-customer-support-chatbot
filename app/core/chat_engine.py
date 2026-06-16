"""Core RAG engine: retrieval → prompt → GPT-4 → response. Supports streaming."""
from __future__ import annotations
import re
from typing import AsyncIterator
from app.core.config import get_settings
from app.core.memory import ConversationMemory
from app.core.logger import get_logger
from app.rag.retriever import Retriever

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a helpful, accurate customer support assistant.
Rules:
- Answer ONLY from the provided context. Say "I don't have that information" if not found.
- Be concise. Use bullet points for multi-step answers.
- Acknowledge frustration before solving.
- Never fabricate prices, policies, or contact details.
- End every reply: "Is there anything else I can help you with?"

Context:
{context}"""

class ChatEngine:
    def __init__(self, retriever: Retriever, memory: ConversationMemory, llm):
        self.retriever = retriever
        self.memory = memory
        self.llm = llm

    def chat(self, user_msg: str) -> str:
        docs = self.retriever.retrieve(user_msg)
        context = "\n\n---\n\n".join(
            [f"[{d.metadata.get('source','?')}]\n{d.content}" for d in docs]) or "No context found."
        self.memory.add("user", user_msg)
        messages = [{"role":"system","content":SYSTEM_PROMPT.format(context=context)}] + self.memory.get_messages()
        try:
            resp = self.llm.chat.completions.create(
                model=settings.llm_model, messages=messages,
                max_tokens=settings.max_tokens, temperature=settings.llm_temperature)
            reply = re.sub(r'\n{3,}', '\n\n', resp.choices[0].message.content.strip())
            self.memory.add("assistant", reply)
            logger.info(f"[{self.memory.session_id}] tokens={resp.usage.total_tokens}")
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm having trouble right now. Please try again shortly."

    async def stream_chat(self, user_msg: str) -> AsyncIterator[str]:
        docs = self.retriever.retrieve(user_msg)
        context = "\n\n---\n\n".join(
            [f"[{d.metadata.get('source','?')}]\n{d.content}" for d in docs]) or "No context found."
        self.memory.add("user", user_msg)
        messages = [{"role":"system","content":SYSTEM_PROMPT.format(context=context)}] + self.memory.get_messages()
        full = ""
        for chunk in self.llm.chat.completions.create(
                model=settings.llm_model, messages=messages,
                max_tokens=settings.max_tokens, temperature=settings.llm_temperature, stream=True):
            delta = chunk.choices[0].delta.content or ""
            full += delta
            yield delta
        self.memory.add("assistant", full)
