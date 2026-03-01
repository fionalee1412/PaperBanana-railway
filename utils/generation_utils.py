# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with kie.ai, Claude, and OpenAI APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import aiofiles
import httpx
from PIL import Image
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# Initialize kie.ai API key
kie_api_key = get_config_val("api_keys", "kie_api_key", "KIE_API_KEY", "")
if kie_api_key:
    print("Loaded KIE API Key for kie.ai")
else:
    print("Warning: Missing KIE_API_KEY. kie.ai calls will fail.")


anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
    print("Initialized Anthropic Client with API Key")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("Initialized OpenAI Client with API Key")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None


# Mapping from short model names to kie.ai image model identifiers
KIE_IMAGE_MODEL_MAP = {
    "nano-banana": "google/nano-banana",
    "google/nano-banana": "google/nano-banana",
    "nano-banana-2": "nano-banana-2",
    "nano-banana-pro": "nano-banana-pro",
    "google/nano-banana-edit": "google/nano-banana-edit",
}


def _convert_to_openai_chat_content(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a generic content list to OpenAI-compatible message content for kie.ai.

    Handles two image formats used across the codebase:
    1. {"type": "image", "source": {"type": "base64", "data": "...", "media_type": "image/jpeg"}}
    2. {"type": "image", "image_base64": "..."}
    """
    result = []
    for item in contents:
        if item.get("type") == "text":
            result.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            # Format 1: source-based (critic / polish agents)
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                result.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
            # Format 2: image_base64 shorthand (planner agent)
            elif item.get("image_base64"):
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                result.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
    return result


async def _call_kie_image_generation(
    model_name: str, prompt: str, aspect_ratio: str = "1:1", image_size: str = "1K"
) -> str:
    """
    Submit an image generation task to kie.ai and poll until completion.
    Returns the base64-encoded image data.
    """
    kie_model = KIE_IMAGE_MODEL_MAP.get(model_name, model_name)

    create_url = "https://api.kie.ai/api/v1/jobs/createTask"
    headers = {"Authorization": f"Bearer {kie_api_key}", "Content-Type": "application/json"}
    body = {
        "model": kie_model,
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": image_size,
            "output_format": "png",
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(create_url, json=body, headers=headers)
        resp.raise_for_status()
        resp_data = resp.json()

    if resp_data.get("code") != 200:
        raise RuntimeError(f"kie.ai createTask failed: {resp_data}")

    task_id = resp_data["data"]["taskId"]
    return await _poll_kie_task(task_id)


async def _poll_kie_task(task_id: str, timeout: float = 300, interval: float = 3) -> str:
    """
    Poll a kie.ai async task until it reaches 'success' or 'fail'.
    Returns the base64 image data on success.
    """
    poll_url = f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}"
    headers = {"Authorization": f"Bearer {kie_api_key}"}
    terminal_states = {"success", "fail"}
    elapsed = 0.0

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < timeout:
            resp = await client.get(poll_url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            state = data.get("state", "")

            if state == "success":
                result_json = json.loads(data.get("resultJson", "{}"))
                urls = result_json.get("resultUrls", [])
                if not urls:
                    raise RuntimeError("kie.ai task succeeded but returned no image URLs")
                return await _download_image_as_base64(urls[0])

            if state == "fail":
                raise RuntimeError(f"kie.ai image generation task failed: {data}")

            await asyncio.sleep(interval)
            elapsed += interval

    raise TimeoutError(f"kie.ai task {task_id} did not complete within {timeout}s")


async def _download_image_as_base64(url: str) -> str:
    """Download an image from a URL and return its base64-encoded content."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call kie.ai API with asynchronous retry logic.
    Routes to image generation or text generation based on config.

    Args:
        model_name: The model identifier (used in the kie.ai URL path for text, or mapped for images).
        contents: Generic content list with text and image items.
        config: A plain dict with keys: system_instruction, temperature, candidate_count,
                max_output_tokens, and optionally response_modalities and image_config.
        max_attempts: Maximum number of retry attempts.
        retry_delay: Base delay in seconds between retries.
        error_context: Optional string for error messages.

    Returns:
        A list of strings (text responses or base64 image data).
    """
    if not kie_api_key:
        raise RuntimeError(
            "KIE API key was not configured. "
            "Please set KIE_API_KEY in environment, or configure api_keys.kie_api_key in configs/model_config.yaml."
        )

    target_candidate_count = config.get("candidate_count", 1)
    response_modalities = config.get("response_modalities", [])
    is_image_generation = "IMAGE" in response_modalities

    # --- Image Generation Path ---
    if is_image_generation:
        image_config = config.get("image_config", {})
        aspect_ratio = image_config.get("aspect_ratio", "1:1")
        image_size = image_config.get("image_size", "1K")

        # Extract the text prompt from contents
        prompt_parts = [item["text"] for item in contents if item.get("type") == "text"]
        prompt = " ".join(prompt_parts)

        result_list = []
        for candidate_idx in range(target_candidate_count):
            for attempt in range(max_attempts):
                try:
                    b64_image = await _call_kie_image_generation(
                        model_name, prompt, aspect_ratio=aspect_ratio, image_size=image_size
                    )
                    result_list.append(b64_image)
                    break
                except Exception as e:
                    context_msg = f" for {error_context}" if error_context else ""
                    current_delay = min(retry_delay * (2 ** attempt), 30)
                    print(
                        f"Attempt {attempt + 1} for image gen model {model_name} "
                        f"(candidate {candidate_idx + 1}) failed{context_msg}: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                    else:
                        print(f"Error: All {max_attempts} attempts failed{context_msg}")
                        result_list.append("Error")

        return result_list

    # --- Text Generation Path ---
    system_instruction = config.get("system_instruction", "")
    temperature = config.get("temperature", 1.0)

    # Build the messages payload
    kie_content = _convert_to_openai_chat_content(contents)
    messages = []
    if system_instruction:
        messages.append({"role": "developer", "content": system_instruction})
    messages.append({"role": "user", "content": kie_content})

    text_url = f"https://api.kie.ai/{model_name}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kie_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "include_thoughts": False,
    }

    async def _single_text_request() -> str:
        """Make a single text completion request with retries."""
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(text_url, json=body, headers=headers)
                    resp.raise_for_status()
                    resp_data = resp.json()
                    text = resp_data["choices"][0]["message"]["content"]
                    if text.strip():
                        return text
                    # Empty response — treat as transient failure
                    raise ValueError("Empty response from kie.ai")
            except Exception as e:
                context_msg = f" for {error_context}" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 30)
                print(
                    f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. "
                    f"Retrying in {current_delay} seconds..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"Error: All {max_attempts} attempts failed{context_msg}")
                    return "Error"
        return "Error"

    # kie.ai returns 1 result per request, so parallelize for multiple candidates
    if target_candidate_count <= 1:
        result = await _single_text_request()
        return [result]

    tasks = [_single_text_request() for _ in range(target_candidate_count)]
    result_list = await asyncio.gather(*tasks)
    return list(result_list)


def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.

    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]

    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                # OpenAI expects data URL format
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }

    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)

            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
