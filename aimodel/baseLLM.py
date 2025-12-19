from langchain_core.language_models.llms import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field
import requests

class VLLM(LLM):
    base_url: str = Field(...)
    api_key: str = Field(...)

    def __call__(self, prompt: str, **kwargs):
        return self._call(prompt, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "prompt": prompt,
            "max_new_tokens": 256,
            "stop": stop if stop else [],
        }
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("completion", "")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"base_url": self.base_url, "api_key": "****"}

    @property
    def _llm_type(self) -> str:
        return "vllm_custom"
