"""OpenAI-compatible chat model (config from .env)."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
)
