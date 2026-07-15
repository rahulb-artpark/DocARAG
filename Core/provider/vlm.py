from dataclasses import dataclass
from itertools import zip_longest
from threading import BoundedSemaphore, Event
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod
from PIL import Image
import base64
from io import BytesIO
from pydantic import BaseModel

from Core.Common.Message import Message
from Core.Common.Memory import Memory
from Core.provider.TokenTracker import TokenTracker

from Core.configs.vlm_config import VLMConfig
import logging
import time
import gc

from Core.utils.bounded_executor import iter_bounded_ordered

log = logging.getLogger(__name__)

GenerationKeyT = TypeVar("GenerationKeyT")
VLMInput = Union[str, List[Dict[str, Any]]]


def _close_controller_client(controller: object) -> None:
    """Release an API controller client across supported SDK versions."""

    client = getattr(controller, "client", None)
    if client is None:
        return
    controller.client = None
    if not getattr(controller, "_owns_client", True):
        return
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    transport = getattr(client, "_client", None)
    close_transport = getattr(transport, "close", None)
    if callable(close_transport):
        close_transport()


def _unload_ollama_model(controller: object) -> None:
    """Evict an Ollama model only when the caller explicitly opted in."""

    if not getattr(controller, "unload_on_close", False):
        return
    client = getattr(controller, "client", None)
    model_name = getattr(controller, "model_name", None)
    if client is None or not model_name:
        return
    try:
        client.generate(model=model_name, prompt="", keep_alive=0)
    except Exception as error:
        log.warning(
            "Ollama VLM unload failed during close (%s)",
            type(error).__name__,
        )


class GenerationAttemptsExhausted(RuntimeError):
    """A single VLM generation did not succeed within its retry budget."""


@dataclass(frozen=True, slots=True)
class GenerationRequest(Generic[GenerationKeyT]):
    """A keyed VLM request for bounded streaming execution."""

    key: GenerationKeyT
    prompt_or_memory: VLMInput
    images: Optional[List[str]] = None


@dataclass(frozen=True, slots=True)
class GenerationResult(Generic[GenerationKeyT]):
    """Typed, per-item VLM generation outcome."""

    index: int
    key: GenerationKeyT
    value: Optional[str] = None
    error: Optional[Exception] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> str:
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise RuntimeError("Successful generation result has no value")
        return self.value


class BatchGenerationError(RuntimeError):
    """One or more items in a compatibility VLM batch failed."""

    def __init__(self, failures: Iterable[GenerationResult[Any]]):
        self.failures = tuple(failures)
        keys = ", ".join(repr(result.key) for result in self.failures[:5])
        if len(self.failures) > 5:
            keys += ", ..."
        super().__init__(
            f"{len(self.failures)} VLM generation request(s) failed"
            + (f" (keys: {keys})" if keys else "")
        )

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BaseVLMController(ABC):
    @abstractmethod
    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        pass


class QwenVLController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
        )
        self.input_device = self._resolve_input_device()
        self.processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._closed = False

    def _resolve_input_device(self):
        """Resolve the loaded model's execution device without assuming CUDA."""

        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except (AttributeError, StopIteration, TypeError) as error:
                raise RuntimeError(
                    "Could not determine the Qwen VLM input device after model load"
                ) from error
        if str(device) == "meta":
            raise RuntimeError("Qwen VLM resolved to the non-executable meta device")
        return device

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the Qwen-VL processor.
        - If prompt_or_memory is a string, creates a new user message with images and text.
        - If it's a list, uses it directly and prepends images to the last user message's content.
        """
        images = images or []

        # --- Case 1: Input is a simple string query ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt_or_memory})
            return [{"role": "user", "content": content}]

        # --- Case 2: Input is a pre-structured list of messages ---
        elif isinstance(prompt_or_memory, list):
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")

            messages = [dict(m) for m in prompt_or_memory]  # Create a copy

            if images:
                last_message = messages[-1]
                if last_message.get("role") != "user":
                    log.warning(
                        "Images can only be added to the last message if it's from the 'user'. Skipping image attachment."
                    )
                    return messages

                # Prepend images to the content of the last user message
                image_content = [{"type": "image", "image": img} for img in images]

                if isinstance(last_message.get("content"), str):
                    # If content is a string, convert it to the list format
                    text_content = [{"type": "text", "text": last_message["content"]}]
                    last_message["content"] = image_content + text_content
                elif isinstance(last_message.get("content"), list):
                    # If content is already a list, prepend the images
                    last_message["content"] = image_content + last_message["content"]
                else:
                    log.warning(
                        f"Unsupported content type in last message: {type(last_message.get('content'))}. Skipping image attachment."
                    )

            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        # Use the helper to prepare messages payload
        messages = self._prepare_messages(prompt_or_memory, images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_device = getattr(self, "input_device", None)
        if input_device is None:
            input_device = self._resolve_input_device()
        inputs = inputs.to(input_device)
        generation_parameters = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_parameters["temperature"] = self.temperature
        generated_ids = self.model.generate(**inputs, **generation_parameters)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images=None,
        schema=None,
    ):
        if schema is None:
            raise ValueError("A Pydantic schema must be provided for generate_json.")
        instruction = (
            "\nReturn only JSON conforming to this schema: "
            f"{schema.model_json_schema()}"
        )
        if isinstance(prompt_or_memory, str):
            prepared_prompt = prompt_or_memory + instruction
        else:
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")
            prepared_prompt = [dict(message) for message in prompt_or_memory]
            last_message = prepared_prompt[-1]
            content = last_message.get("content")
            if isinstance(content, str):
                last_message["content"] = content + instruction
            elif isinstance(content, list):
                last_message["content"] = list(content) + [
                    {"type": "text", "text": instruction}
                ]
            else:
                raise ValueError("Last message content must be text or content parts")
        generated = self.generate(prepared_prompt, images)
        return schema.model_validate_json(generated)

    def close(self) -> None:
        """Drop in-process model weights and return cached CUDA memory."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # impossible for a normally constructed controller
            pass


class GPTVLMController(BaseVLMController):
    def __init__(
        self,
        config: VLMConfig,
        *,
        client: Optional[object] = None,
        owns_client: Optional[bool] = None,
    ):
        if owns_client is not None and not isinstance(owns_client, bool):
            raise ValueError("owns_client must be a boolean when provided")
        self.model_name = config.model_name or "gpt-4o"
        self.request_timeout_seconds = config.request_timeout_seconds
        if self.request_timeout_seconds is None:  # guarded by VLMConfig
            raise ValueError("remote GPT VLM requires request_timeout_seconds")
        if client is None:
            from openai import OpenAI

            parameters = {
                "api_key": config.api_key,
                "timeout": self.request_timeout_seconds,
                # VLM owns its retry loop; disable the SDK's additional retry
                # multiplier so the configured latency bound is truthful.
                "max_retries": 0,
            }
            if config.api_base:
                parameters["base_url"] = config.api_base
            self.client = OpenAI(**parameters)
            self._owns_client = True if owns_client is None else owns_client
        else:
            self.client = client
            self._owns_client = False if owns_client is None else owns_client
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._closed = False

    def _encode_image(self, image_path):
        if isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format="JPEG")
            img_data = buffered.getvalue()
            base64_encoded = base64.b64encode(img_data).decode("utf-8")
            return base64_encoded
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, Memory],
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the OpenAI API call.
        - If prompt_or_memory is a string, creates a new user message with multimodal content.
        - If it's a list, it uses the list directly and intelligently adds images to the last message.
        """
        # --- Case 1: Input is a simple string query ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img in images:
                    base64_image = self._encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )
            return [{"role": "user", "content": content}]

        # --- Case 2: Input is a pre-structured list of messages ---
        elif isinstance(prompt_or_memory, (Memory, list)):
            if isinstance(prompt_or_memory, Memory):
                messages = [
                    {"role": message.role, "content": message.content}
                    for message in prompt_or_memory.get()
                ]
            else:
                if any(not isinstance(message, dict) for message in prompt_or_memory):
                    raise TypeError("Each message must be a dictionary")
                messages = [dict(message) for message in prompt_or_memory]
            if not messages:
                raise ValueError("Message list cannot be empty.")

            # If images are provided, find the last message and append images to its content
            if images:
                last_message = messages[-1]
                # Add a check to ensure images are only added to a 'user' message,
                # as per the OpenAI API specification.
                if last_message.get("role") == "user":
                    # This internal logic was already correct.
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = [
                            {"type": "text", "text": last_message["content"]}
                        ]
                    elif last_message.get("content") is None:
                        last_message["content"] = []

                    if isinstance(last_message["content"], list):
                        for img in images:
                            base64_image = self._encode_image(img)
                            last_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
                            )
                    else:
                        log.warning(
                            "Could not attach images: The 'content' of the last message is not a string or list."
                        )
                else:
                    log.warning(
                        f"Could not attach images: The last message role is '{last_message.get('role')}', not 'user'."
                    )
                # --- MODIFICATION 2 END ---
            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        content = self._prepare_messages(prompt_or_memory, images)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout_seconds,
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        return completion.choices[0].message.content

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        schema: Type[BaseModel] = None,
    ) -> Dict:
        if not schema:
            raise ValueError("A Pydantic schema must be provided for generate_json.")

        # Prepare messages first, which might include a system prompt
        messages = self._prepare_messages(prompt_or_memory, images)

        # Add or modify the system prompt to include JSON instructions
        json_instruction = f"\nYour output MUST conform to this JSON schema: {schema.model_json_schema()}"

        # Check if a system message already exists
        system_message_exists = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] += json_instruction
                system_message_exists = True
                break

        if not system_message_exists:
            messages.insert(0, {"role": "system", "content": json_instruction})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout_seconds,
            response_format={"type": "json_object"},  # Use modern JSON mode
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
            log.info(
                f"Prompt tokens: {completion.usage.prompt_tokens}, Completion tokens: {completion.usage.completion_tokens}"
            )

        return schema.model_validate_json(completion.choices[0].message.content)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _close_controller_client(self)


class OllamaVLMController(BaseVLMController):
    def __init__(
        self,
        config: VLMConfig,
        *,
        client: Optional[object] = None,
        owns_client: Optional[bool] = None,
    ):
        if owns_client is not None and not isinstance(owns_client, bool):
            raise ValueError("owns_client must be a boolean when provided")
        self.model_name = config.model_name or "qwen2.5vl:latest"
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.keep_alive = config.keep_alive
        self.unload_on_close = config.unload_on_close
        self.request_timeout_seconds = config.request_timeout_seconds
        if self.request_timeout_seconds is None:  # guarded by VLMConfig
            raise ValueError("remote Ollama VLM requires request_timeout_seconds")
        if client is None:
            import ollama

            parameters: Dict[str, Any] = {
                "host": config.api_base or "http://127.0.0.1:11434",
                "timeout": self.request_timeout_seconds,
            }
            if config.api_key:
                parameters["headers"] = {
                    "Authorization": f"Bearer {config.api_key}"
                }
            self.client = ollama.Client(**parameters)
            self._owns_client = True if owns_client is None else owns_client
        else:
            self.client = client
            self._owns_client = False if owns_client is None else owns_client
        self._closed = False

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the Ollama API call.
        - If prompt_or_memory is a string, creates a new user message.
        - If it's a list of dicts, uses it directly and appends images to the last message if needed.
        """
        if isinstance(prompt_or_memory, str):
            # Case 1: Input is a simple string query.
            log.debug("Preparing messages from a string query.")
            return [
                {"role": "user", "content": prompt_or_memory, "images": images or []}
            ]

        elif isinstance(prompt_or_memory, list):
            # Case 2: Input is a pre-structured list of messages.
            log.debug("Using pre-structured message list.")
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")

            # Make a copy to avoid modifying the original list passed by the caller.
            messages = [dict(m) for m in prompt_or_memory]

            # If images are provided, add them to the last message if it doesn't have images already.
            if images:
                last_message = messages[-1]
                if last_message.get("role") == "user" and not last_message.get(
                    "images"
                ):
                    log.debug("Attaching images to the last user message.")
                    last_message["images"] = images
                else:
                    log.warning(
                        "Could not attach images: Last message is not a user role or already contains images."
                    )

            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        images = images or []
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                keep_alive=self.keep_alive,
            )

            if response:
                tracker = TokenTracker.get_instance()
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                try:
                    prompt_tokens = int(prompt_tokens)
                except (TypeError, ValueError):
                    logging.warning("Ollama returned an invalid prompt token count")
                    prompt_tokens = 0

                try:
                    completion_tokens = int(completion_tokens)
                except (TypeError, ValueError):
                    logging.warning("Ollama returned an invalid completion token count")
                    completion_tokens = 0
                tracker.add_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"Could not get a response from Ollama model '{self.model_name}'"
            ) from e

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        images = images or []
        if schema is None:
            raise ValueError("A Pydantic schema must be provided for generate_json.")
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format=schema.model_json_schema(),
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                keep_alive=self.keep_alive,
            )

            if response:
                tracker = TokenTracker.get_instance()
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                try:
                    prompt_tokens = int(prompt_tokens)
                except (TypeError, ValueError):
                    logging.warning("Ollama returned an invalid prompt token count")
                    prompt_tokens = 0

                try:
                    completion_tokens = int(completion_tokens)
                except (TypeError, ValueError):
                    logging.warning("Ollama returned an invalid completion token count")
                    completion_tokens = 0
                tracker.add_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            return schema.model_validate_json(response["message"]["content"])
        except Exception as e:
            raise RuntimeError(
                f"Could not get a JSON response from Ollama model '{self.model_name}'"
            ) from e

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _unload_ollama_model(self)
        _close_controller_client(self)


class VLM:
    def __init__(self, vlm_config: Optional[Dict[str, Any]] = None):
        if vlm_config is None:
            raise ValueError("VLM config must be provided")
        if not isinstance(vlm_config, VLMConfig):
            config = VLMConfig(**vlm_config)
        else:
            config = vlm_config
        self.config = config
        self._closed = False
        self.max_workers = config.max_workers
        self.max_in_flight = config.max_in_flight
        backend = config.backend.lower()
        provider_concurrency = 1 if backend == "qwen" else config.max_workers
        self._admission = BoundedSemaphore(provider_concurrency)
        if backend == "qwen":
            self.vlm = QwenVLController(config)
        elif backend == "gpt":
            self.vlm = GPTVLMController(config)
        elif backend == "ollama":
            self.vlm = OllamaVLMController(config)
        else:
            raise ValueError(f"Unsupported VLM backend: {backend}")

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                admission = getattr(self, "_admission", None)
                if admission is None:
                    response = self.vlm.generate(prompt_or_memory, images)
                else:
                    with admission:
                        response = self.vlm.generate(prompt_or_memory, images)
                if not isinstance(response, str) or not response.strip():
                    raise ValueError("Empty response from VLM")
                return response
            except Exception as e:
                last_exception = e
                log.warning(
                    "VLM generation attempt %d/%d failed (%s)",
                    attempt,
                    self.config.max_retries,
                    type(e).__name__,
                )
                if (
                    attempt < self.config.max_retries
                    and self.config.retry_delay_seconds
                ):
                    time.sleep(self.config.retry_delay_seconds)
        raise GenerationAttemptsExhausted(
            f"VLM generation failed after {self.config.max_retries} attempt(s)"
        ) from last_exception

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                admission = getattr(self, "_admission", None)
                if admission is None:
                    response = self.vlm.generate_json(prompt_or_memory, images, schema)
                else:
                    with admission:
                        response = self.vlm.generate_json(
                            prompt_or_memory, images, schema
                        )
                if not response:
                    raise ValueError("Empty JSON response from VLM")
                return response
            except Exception as e:
                last_exception = e
                log.warning(
                    "VLM JSON generation attempt %d/%d failed (%s)",
                    attempt,
                    self.config.max_retries,
                    type(e).__name__,
                )
                if (
                    attempt < self.config.max_retries
                    and self.config.retry_delay_seconds
                ):
                    time.sleep(self.config.retry_delay_seconds)
        raise GenerationAttemptsExhausted(
            f"VLM JSON generation failed after {self.config.max_retries} attempt(s)"
        ) from last_exception

    def iter_generation_results(
        self,
        requests: Iterable[GenerationRequest[GenerationKeyT]],
        *,
        cancel_event: Optional[Event] = None,
        max_workers: Optional[int] = None,
        max_in_flight: Optional[int] = None,
    ) -> Iterator[GenerationResult[GenerationKeyT]]:
        """Yield ordered VLM outcomes with bounded request submission."""

        worker_limit = self.max_workers if max_workers is None else max_workers
        if isinstance(self.vlm, QwenVLController):
            # The in-process Qwen model is not thread-safe, but it can process a
            # large request stream safely when serialized.
            worker_limit = 1
        in_flight_limit = (
            self.max_in_flight if max_in_flight is None else max_in_flight
        )

        def generate(request: GenerationRequest[GenerationKeyT]) -> str:
            return self.generate(request.prompt_or_memory, request.images)

        for outcome in iter_bounded_ordered(
            requests,
            generate,
            max_workers=worker_limit,
            max_in_flight=in_flight_limit,
            cancel_event=cancel_event,
            thread_name_prefix="vlm-generation",
        ):
            yield GenerationResult(
                index=outcome.index,
                key=outcome.item.key,
                value=outcome.value,
                error=outcome.error,
            )

    def batch_generate(
        self,
        queries: Iterable[VLMInput],
        images_list: Optional[Iterable[Optional[Union[str, List[str]]]]] = None,
        max_workers: Optional[int] = None,
        max_in_flight: Optional[int] = None,
    ) -> List[str]:
        """Return ordered strings for legacy callers, raising on any failure."""

        def requests() -> Iterator[GenerationRequest[int]]:
            if images_list is None:
                for index, query in enumerate(queries):
                    yield GenerationRequest(key=index, prompt_or_memory=query)
                return

            missing = object()
            for index, pair in enumerate(
                zip_longest(queries, images_list, fillvalue=missing)
            ):
                query, images = pair
                if query is missing or images is missing:
                    raise ValueError("queries and images_list must have equal lengths")
                if isinstance(images, str):
                    normalized_images: Optional[List[str]] = [images]
                elif images is None:
                    normalized_images = None
                else:
                    normalized_images = list(images)
                yield GenerationRequest(
                    key=index,
                    prompt_or_memory=query,
                    images=normalized_images,
                )

        results: List[str] = []
        failures: List[GenerationResult[int]] = []
        for outcome in self.iter_generation_results(
            requests(),
            max_workers=max_workers,
            max_in_flight=max_in_flight,
        ):
            if outcome.ok:
                results.append(outcome.unwrap())
            else:
                failures.append(outcome)
        if failures:
            raise BatchGenerationError(failures)
        return results

    def close(self) -> None:
        """Release the selected VLM controller idempotently."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        controller = getattr(self, "vlm", None)
        self.vlm = None
        if controller is None:
            return
        close = getattr(controller, "close", None)
        if callable(close):
            close()
        else:  # compatibility with injected/older controllers
            _close_controller_client(controller)


if __name__ == "__main__":
    vlm_config = VLMConfig()
    vlm = VLM(vlm_config)
    # llm = LLM("Qwen/Qwen2.5-VL-7B-Instruct")
    # llm = LLM('gpt-4o')

    tmp_memory = Memory()
    query = (
        "Description this image in a senence, and then list the objects in the image."
    )
    sys_temp = "You are a helpful assistant that helps people find information."
    tmp_memory.add(Message(role="system", content=sys_temp))
    tmp_memory.add(Message(role="user", content=query))
    response = vlm.generate(
        prompt_or_memory=tmp_memory,
        images=[
            "/home/wangshu/multimodal/GBC-RAG/test/tree_index/images/8f4d58edc0302540d157aa54eaabfddf7534f4b407d4c811993b60372678a274.jpg"
        ],
    )
    print(response)
