from dataclasses import dataclass
from threading import BoundedSemaphore, Event
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel
from Core.Common.Memory import Memory
from Core.configs.llm_config import LLMConfig
from Core.utils.bounded_executor import iter_bounded_ordered
from Core.utils.utils import get_max_output_tokens
from Core.provider.TokenTracker import TokenTracker
from Core.provider.structured_output import parse_structured
import time

log = logging.getLogger(__name__)

CompletionKeyT = TypeVar("CompletionKeyT")


def _close_controller_client(controller: object) -> None:
    """Release a controller's HTTP client without depending on its SDK type.

    OpenAI exposes ``close`` on its public client.  Some Ollama releases wrap
    an httpx client without exposing that method, so the guarded fallback is
    used only when no public close operation exists.  References are broken
    before invoking close so repeated shutdown remains harmless even when the
    SDK raises.
    """

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
    """Best-effort opt-in eviction of a model from a shared Ollama server."""

    if not getattr(controller, "unload_on_close", False):
        return
    client = getattr(controller, "client", None)
    model = getattr(controller, "model", None)
    if client is None or not model:
        return
    try:
        client.generate(model=model, prompt="", keep_alive=0)
    except Exception as error:  # cleanup must still release the HTTP client
        log.warning(
            "Ollama model unload failed during close (%s)",
            type(error).__name__,
        )


class CompletionAttemptsExhausted(RuntimeError):
    """A single completion did not succeed within its retry budget."""


@dataclass(frozen=True, slots=True)
class CompletionRequest(Generic[CompletionKeyT]):
    """A keyed completion request used by streaming ingestion callers."""

    key: CompletionKeyT
    prompt: Union[str, Memory]


@dataclass(frozen=True, slots=True)
class CompletionResult(Generic[CompletionKeyT]):
    """Typed, per-item completion outcome."""

    index: int
    key: CompletionKeyT
    value: Optional[str] = None
    error: Optional[Exception] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> str:
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise RuntimeError("Successful completion result has no value")
        return self.value


class BatchCompletionError(RuntimeError):
    """One or more items in a compatibility batch failed."""

    def __init__(self, failures: Iterable[CompletionResult[Any]]):
        self.failures = tuple(failures)
        keys = ", ".join(repr(result.key) for result in self.failures[:5])
        if len(self.failures) > 5:
            keys += ", ..."
        super().__init__(
            f"{len(self.failures)} completion request(s) failed"
            + (f" (keys: {keys})" if keys else "")
        )


class BaseLLMController(ABC):
    @abstractmethod
    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list in the format required by the specific API.
        This must be implemented by each subclass.
        """
        pass

    @abstractmethod
    def get_completion(self, prompt_or_memory: Union[str, Memory]) -> str:
        """Get completion from LLM."""
        pass

    @abstractmethod
    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[List[str]] = None,
    ) -> dict:
        """Get structured JSON response from LLM using Pydantic schema."""
        pass


class OpenAIController(BaseLLMController):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        *,
        client: Optional[object] = None,
        owns_client: Optional[bool] = None,
    ):
        if owns_client is not None and not isinstance(owns_client, bool):
            raise ValueError("owns_client must be a boolean when provided")
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "gpt-3.5-turbo"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        base_url = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )
        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )
        self.request_timeout_seconds = (
            llm_config.request_timeout_seconds if llm_config else 300.0
        )

        if client is None:
            from openai import OpenAI

            parameters = {
                "timeout": self.request_timeout_seconds,
                # The LLM wrapper owns the explicit retry budget. Disabling
                # hidden SDK retries keeps the configured attempt count exact;
                # SDK timeouts use their underlying transport semantics.
                "max_retries": 0,
            }
            api_key = llm_config.api_key if llm_config is not None else None
            if api_key is not None:
                parameters["api_key"] = api_key
            if base_url is not None:
                parameters["base_url"] = base_url
            self.client = OpenAI(**parameters)
            self._owns_client = True if owns_client is None else owns_client
        else:
            self.client = client
            self._owns_client = False if owns_client is None else owns_client
        self._closed = False

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepares messages for the OpenAI API format."""
        if isinstance(prompt_or_memory, Memory):
            res_dict = [msg.to_dict() for msg in prompt_or_memory.storage]
            return res_dict

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        parameters = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": get_max_output_tokens(messages, self.max_tokens),
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.request_timeout_seconds,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if json_response:
            parameters["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**parameters)

        if response.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        Get structured JSON response from OpenAI using Pydantic schema.
        :param prompt: str
        :param schema: Pydantic BaseModel
        :param images: Optional[list] = None
        :return: dict
        """
        if isinstance(prompt_or_memory, Memory):
            # If it's a memory object, use its history.
            # We will attach images to the content of the *last* message.
            messages = [msg.to_dict() for msg in prompt_or_memory.storage]
            last_content = messages[-1]["content"]

            content_list = [{"type": "text", "text": last_content}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages[-1]["content"] = content_list

        else:  # It's a simple string prompt
            content_list = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages = [{"role": "user", "content": content_list}]

        completion = self.client.beta.chat.completions.parse(
            temperature=self.temperature,
            model=self.model,
            messages=messages,
            max_tokens=get_max_output_tokens(messages, self.max_tokens),
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            timeout=self.request_timeout_seconds,
            response_format=schema,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": think_mode},
            },
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        message = completion.choices[0].message
        if hasattr(message, "parsed") and message.parsed:
            return message.parsed
        elif hasattr(message, "refusal") and message.refusal:
            return {"refusal": message.refusal}
        else:
            return {}

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _close_controller_client(self)


class OllamaController(BaseLLMController):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        *,
        client: Optional[object] = None,
        owns_client: Optional[bool] = None,
    ):
        if owns_client is not None and not isinstance(owns_client, bool):
            raise ValueError("owns_client must be a boolean when provided")
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "llama3.1"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )
        self.api_base = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )
        self.api_key = (
            llm_config.api_key
            if llm_config and hasattr(llm_config, "api_key")
            else None
        )
        self.keep_alive = llm_config.keep_alive if llm_config else "5m"
        self.num_ctx = getattr(llm_config, "num_ctx", None) if llm_config else None
        self.unload_on_close = bool(llm_config and llm_config.unload_on_close)
        self.request_timeout_seconds = (
            llm_config.request_timeout_seconds if llm_config else 300.0
        )
        if client is None:
            import ollama

            parameters: Dict[str, Any] = {
                "host": self.api_base,
                "timeout": self.request_timeout_seconds,
            }
            if self.api_key:
                parameters["headers"] = {
                    "Authorization": f"Bearer {self.api_key}"
                }
            self.client = ollama.Client(**parameters)
            self._owns_client = True if owns_client is None else owns_client
        else:
            self.client = client
            self._owns_client = False if owns_client is None else owns_client
        self._closed = False

    def _options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepares messages for the Ollama API format."""
        if isinstance(prompt_or_memory, Memory):
            return prompt_or_memory.get()

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        parameters = {
            "model": self.model,
            "messages": messages,
            "options": self._options(),
            "keep_alive": self.keep_alive,
            "think": False,
        }
        if json_response:
            parameters["format"] = "json"
        response = self.client.chat(**parameters)

        if response:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )

        return response["message"]["content"]

    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        Get structured JSON response from Ollama using Pydantic schema.
        :param prompt: str
        :param schema: Pydantic BaseModel
        :param images: Optional[list] = None
        :return: dict
        """
        messages = self._prepare_messages(prompt_or_memory)

        if images:
            # Ollama expects images at the top level of the message dictionary
            messages[-1]["images"] = images
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),  # 关键：传递schema
            options=self._options(),
            keep_alive=self.keep_alive,
            think=think_mode,
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

        return parse_structured(response["message"]["content"], schema)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _unload_ollama_model(self)
        _close_controller_client(self)


class LLM:
    """LLM-based controller for memory metadata generation"""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
    ):
        if llm_config is None:
            raise ValueError("Config must be provided")
        if not isinstance(llm_config, LLMConfig):
            config = LLMConfig(**llm_config)
        else:
            config = llm_config
        self.config: LLMConfig = config
        self._closed = False
        self.max_workers = config.max_workers
        self.max_in_flight = config.max_in_flight
        self._admission = BoundedSemaphore(config.max_workers)
        backend = config.backend
        if backend == "openai":
            self.llm = OpenAIController(llm_config=config)
        elif backend == "ollama":
            self.llm = OllamaController(llm_config=config)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")

    def get_completion(
        self, prompt: Union[str, Memory], json_response: bool = False
    ) -> str:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                admission = getattr(self, "_admission", None)
                if admission is None:
                    res = self.llm.get_completion(prompt, json_response)
                else:
                    with admission:
                        res = self.llm.get_completion(prompt, json_response)
                if not isinstance(res, str) or not res.strip():
                    raise ValueError("Empty response from LLM")
                return res
            except Exception as e:
                last_exception = e
                log.warning(
                    "LLM completion attempt %d/%d failed (%s)",
                    attempt,
                    self.config.max_retries,
                    type(e).__name__,
                )
                if (
                    attempt < self.config.max_retries
                    and self.config.retry_delay_seconds
                ):
                    time.sleep(self.config.retry_delay_seconds)
        raise CompletionAttemptsExhausted(
            f"LLM completion failed after {self.config.max_retries} attempt(s)"
        ) from last_exception

    def iter_completion_results(
        self,
        requests: Iterable[CompletionRequest[CompletionKeyT]],
        json_response: bool = False,
        *,
        cancel_event: Optional[Event] = None,
        max_in_flight: Optional[int] = None,
    ) -> Iterator[CompletionResult[CompletionKeyT]]:
        """Yield ordered outcomes while applying backpressure to ``requests``.

        Unlike :meth:`batch_get_completion`, this method does not allocate a
        result list and never turns an exception into response text.  It is the
        preferred API for document-ingestion pipelines.
        """

        in_flight_limit = (
            self.max_in_flight if max_in_flight is None else max_in_flight
        )

        def complete(request: CompletionRequest[CompletionKeyT]) -> str:
            return self.get_completion(request.prompt, json_response)

        for outcome in iter_bounded_ordered(
            requests,
            complete,
            max_workers=self.max_workers,
            max_in_flight=in_flight_limit,
            cancel_event=cancel_event,
            thread_name_prefix="llm-completion",
        ):
            yield CompletionResult(
                index=outcome.index,
                key=outcome.item.key,
                value=outcome.value,
                error=outcome.error,
            )

    def iter_batch_get_completion(
        self,
        prompts: Iterable[Union[str, Memory]],
        json_response: bool = False,
        *,
        cancel_event: Optional[Event] = None,
        max_in_flight: Optional[int] = None,
    ) -> Iterator[CompletionResult[int]]:
        """Compatibility-friendly typed stream keyed by prompt position."""

        requests = (
            CompletionRequest(key=index, prompt=prompt)
            for index, prompt in enumerate(prompts)
        )
        return self.iter_completion_results(
            requests,
            json_response=json_response,
            cancel_event=cancel_event,
            max_in_flight=max_in_flight,
        )

    def batch_get_completion(
        self, prompts: Iterable[Union[str, Memory]], json_response: bool = False
    ) -> List[str]:
        """Return ordered strings for legacy callers, raising on any failure."""

        results: List[str] = []
        failures: List[CompletionResult[int]] = []
        for outcome in self.iter_batch_get_completion(prompts, json_response):
            if outcome.ok:
                results.append(outcome.unwrap())
            else:
                failures.append(outcome)
        if failures:
            raise BatchCompletionError(failures)
        return results

    def get_json_completion(
        self,
        prompt: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                admission = getattr(self, "_admission", None)
                if admission is None:
                    res = self.llm.get_json_completion(
                        prompt, schema, images, think_mode=think_mode
                    )
                else:
                    with admission:
                        res = self.llm.get_json_completion(
                            prompt, schema, images, think_mode=think_mode
                        )
                if not res:
                    raise ValueError("Empty response from LLM")
                return res
            except Exception as e:
                last_exception = e
                log.warning(
                    "LLM JSON completion attempt %d/%d failed (%s)",
                    attempt,
                    self.config.max_retries,
                    type(e).__name__,
                )
                if (
                    attempt < self.config.max_retries
                    and self.config.retry_delay_seconds
                ):
                    time.sleep(self.config.retry_delay_seconds)
        raise CompletionAttemptsExhausted(
            f"LLM JSON completion failed after {self.config.max_retries} attempt(s)"
        ) from last_exception

    def close(self) -> None:
        """Release the selected controller and its network client idempotently."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        controller = getattr(self, "llm", None)
        self.llm = None
        if controller is None:
            return
        close = getattr(controller, "close", None)
        if callable(close):
            close()
        else:  # compatibility with injected/older controllers
            _close_controller_client(controller)


if __name__ == "__main__":
    # Example usage
    print("Testing LLM controller...")

    # from Core.configs.system_config import load_system_config

    # cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")
    # cfg.llm.max_workers = 4
    # controller = LLM(llm_config=cfg.llm)
    # prompt = "respond in 20 words. who are you?"

    # token_tracker = TokenTracker.get_instance()
    # token_tracker.reset()

    # start_time = time.time()
    # response = controller.batch_get_completion(prompts=[prompt] * 8)
    # end_time = time.time()
    # print(response)
    # print(f"Response time: {end_time - start_time:.2f} seconds")

    # batch_cost = token_tracker.record_stage("batch_get_completion")
    # print(f"Batch cost: {batch_cost}")

    # tmp_memory = Memory()
    # tmp_memory.add_batch(
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What is the capital of France?"},
    #     ]
    # )
    # response = controller.get_completion(tmp_memory)
    # print(response)
    # memory_cost = token_tracker.record_stage("get_completion")
    # print(f"Memory cost: {memory_cost}")
    # token_tracker.print_all_stages()
