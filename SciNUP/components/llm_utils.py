"""Utility functions for interacting with LLMs.

Set your API key is as an environment variable by running the following command
in your terminal:
export OPENROUTER_API_KEY='your-api-key-here'
"""

import os
from typing import Optional

import litellm
from ollama import Client

OLLAMA_HOST = "https://ollama.ux.uis.no"


def call_llm(
    prompt: str,
    model: str,
    backend: str = "openrouter",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_base: str = "https://openrouter.ai/api/v1",
    temperature: float = 0,
) -> str:
    """
    Calls a large language model with the given prompt and parameters.

    Args:
        prompt: The input prompt for the LLM.
        model: The name of the model to use (e.g.,
            'openrouter/meta-llama/llama-4-maverick-17b-128e-instruct:free').
        backend: The backend to use for LLM inference ('openrouter' or 'ollama').
        system_prompt: An optional instruction that defines the LLM's
            persona or behavior.
        max_tokens: The maximum number of tokens to generate.
        api_base: The base URL for the LLM API.
        temperature: Temperature parameter for LLM, default is 0.

    Returns:
        The content of the LLM's response as a string.

    Raises:
        ValueError: If the OPENROUTER_API_KEY environment variable is not set.
    """
    if backend == "ollama":
        try:
            client = Client(host=OLLAMA_HOST)
            final_prompt = (
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            )
            response = client.generate(
                model=model,
                prompt=final_prompt,
            )
            return response["response"].strip()
        except Exception as e:
            print(f"An error occurred while calling Ollama: {e}")
            return ""
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        # Dynamically build the messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=model,
                api_key=api_key,
                api_base=api_base,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_text = response.choices[0].message.content.strip()
            return response_text

        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return ""


def get_last_response_line(response: str) -> str:
    """Extracts the last non-empty line from the LLM response."""
    lines = [
        line.strip() for line in response.strip().split("\n") if line.strip()
    ]
    return lines[-1] if lines else ""
