import os
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 60.0
_DEFAULT_MAX_TOKENS = 32768

DEEPSEEK_API_KEY = "REDACTED_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"

_REASONING_MODELS = ("deepseek-reasoner", "o1", "o3", "r1")


class LLMClient:
    def __init__(
        self,
        model_name: str = DEEPSEEK_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        from openai import OpenAI
        _api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY") or DEEPSEEK_API_KEY
        _base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("BASE_URL") or DEEPSEEK_BASE_URL
        self.client = OpenAI(api_key=_api_key, base_url=_base_url)
        logger.info(f"LLM client initialized: model={model_name}, base_url={_base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        is_reasoning = any(k in self.model_name.lower() for k in _REASONING_MODELS)
        extra_kwargs = (
            {"reasoning_effort": "high"} if is_reasoning
            else {"extra_body": {"thinking": {"type": "disabled"}}}
        )

        last_exc: Exception = RuntimeError("No attempt made")
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    **extra_kwargs,
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if not content or not content.strip():
                    raise ValueError(f"Empty response (finish_reason={finish_reason!r})")
                return content.strip()
            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
                    logger.warning(f"API call failed (attempt {attempt+1}/{_MAX_RETRIES}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"API call failed after {_MAX_RETRIES} attempts: {e}")
        raise last_exc


def create_llm_client(
    model_name: str = DEEPSEEK_DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> LLMClient:
    return LLMClient(model_name=model_name, api_key=api_key, base_url=base_url,
                     temperature=temperature, max_tokens=max_tokens)
