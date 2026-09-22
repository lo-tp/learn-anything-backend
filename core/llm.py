"""OpenAI-compatible chat model (config from .env)."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.mock_llm import is_mock_mode

load_dotenv()

if is_mock_mode():
    # #118: mock mode must start without OPENAI_API_KEY. A real ChatOpenAI is
    # still constructed (so the material graph stays wired to the real llm)
    # with a placeholder key — it is never invoked in mock mode.
    _api_key = SecretStr("mock")
else:
    _api_key = SecretStr(os.getenv("OPENAI_API_KEY", ""))

llm = ChatOpenAI(
    api_key=_api_key,
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    max_tokens=65536,  # type: ignore[call-arg]
)
