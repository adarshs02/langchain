"""`Runnable` that routes between models based on semantic confidence (logprobs)."""

import math
from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    coro_with_context,
    get_unique_config_specs,
)


def _require_accumulated(
    accumulated: AIMessageChunk | None,
) -> AIMessageChunk:
    """Validate that the primary model produced at least one chunk."""
    if accumulated is None:
        msg = "Primary model yielded no chunks"
        raise ValueError(msg)
    return accumulated


class TieredSemanticRouter(RunnableSerializable[LanguageModelInput, AIMessage]):
    """Route between a fast primary model and a stronger fallback based on confidence.

    Unlike `RunnableWithFallbacks` which triggers on exceptions, this router
    invokes the primary model first and inspects output confidence via logprobs.
    If the average logprob falls below a threshold, the input is automatically
    re-sent to a stronger fallback model.

    The primary model is always called with `logprobs=True` so that token-level
    log-probabilities are available in `response_metadata`.

    Example:
        ```python
        from langchain_core.runnables import TieredSemanticRouter

        router = TieredSemanticRouter(
            primary=fast_model,
            fallback=strong_model,
            threshold=-1.0,
        )
        result = router.invoke("Explain quantum entanglement.")
        ```

    !!! warning
        Streaming collects the full primary response before deciding whether
        to escalate, so there is no incremental output until the primary
        completes. Users needing true incremental streaming should use
        `RunnableWithFallbacks` instead.
    """

    primary: BaseChatModel
    """Fast / cheap model tried first."""
    fallback: BaseChatModel
    """Stronger model used when the primary is uncertain."""
    threshold: float = Field(default=-1.0)
    """Average logprob cutoff.

    `0.0` means maximum confidence; more-negative values are less confident.
    Primary results with an average logprob `>= threshold` are returned
    directly; otherwise the fallback model is invoked.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for step in [self.primary, self.fallback] for spec in step.config_specs
        )

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(message: AIMessage) -> float:
        """Compute average logprob from a model response.

        Args:
            message: An `AIMessage` whose `response_metadata` contains
                logprob data under
                `response_metadata["logprobs"]["content"]`.

        Returns:
            The arithmetic mean of all finite token logprobs.

        Raises:
            ValueError: If logprobs are absent or yield no usable values.
        """
        logprobs_data = (message.response_metadata or {}).get("logprobs")
        if not logprobs_data:
            msg = (
                "Primary model response does not contain logprobs. "
                "Ensure the model supports the logprobs parameter."
            )
            raise ValueError(msg)

        content = logprobs_data.get("content")
        if not content:
            msg = (
                "Primary model response contains empty logprobs "
                "content. Cannot calculate confidence."
            )
            raise ValueError(msg)

        values = [
            t["logprob"]
            for t in content
            if t.get("logprob") is not None and math.isfinite(t["logprob"])
        ]
        if not values:
            msg = (
                "No finite logprob values found in primary model "
                "response. Cannot calculate confidence."
            )
            raise ValueError(msg)

        return float(sum(values) / len(values))

    # ------------------------------------------------------------------
    # invoke / ainvoke
    # ------------------------------------------------------------------

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input if isinstance(input, dict) else {"input": input},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                primary_result = context.run(
                    self.primary.invoke,
                    input,
                    config,
                    logprobs=True,
                    **kwargs,
                )

            confidence = self._calculate_confidence(primary_result)
            if confidence >= self.threshold:
                run_manager.on_chain_end(primary_result)
                return primary_result

            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                fallback_result = context.run(
                    self.fallback.invoke,
                    input,
                    config,
                    **kwargs,
                )
            run_manager.on_chain_end(fallback_result)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        return fallback_result

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input if isinstance(input, dict) else {"input": input},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                coro = context.run(
                    self.primary.ainvoke,
                    input,
                    config,
                    logprobs=True,
                    **kwargs,
                )
                primary_result = await coro_with_context(coro, context)

            confidence = self._calculate_confidence(primary_result)
            if confidence >= self.threshold:
                await run_manager.on_chain_end(primary_result)
                return primary_result

            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                coro = context.run(
                    self.fallback.ainvoke,
                    input,
                    config,
                    **kwargs,
                )
                fallback_result = await coro_with_context(coro, context)
            await run_manager.on_chain_end(fallback_result)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        return fallback_result

    # ------------------------------------------------------------------
    # stream / astream  (collect-then-decide)
    # ------------------------------------------------------------------

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input if isinstance(input, dict) else {"input": input},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            # Collect all primary chunks first
            child_config = patch_config(config, callbacks=run_manager.get_child())
            chunks: list[AIMessageChunk] = []
            accumulated: AIMessageChunk | None = None
            with set_config_context(child_config) as context:
                stream = context.run(
                    self.primary.stream,
                    input,
                    config,
                    logprobs=True,
                    **kwargs,
                )
                for chunk in stream:
                    chunks.append(chunk)
                    accumulated = chunk if accumulated is None else accumulated + chunk

            accumulated = _require_accumulated(accumulated)
            confidence = self._calculate_confidence(accumulated)
            if confidence >= self.threshold:
                for chunk in chunks:
                    yield chunk
                run_manager.on_chain_end(accumulated)
                return

            # Fallback: stream from the stronger model
            child_config = patch_config(config, callbacks=run_manager.get_child())
            fallback_output: AIMessageChunk | None = None
            with set_config_context(child_config) as context:
                stream = context.run(
                    self.fallback.stream,
                    input,
                    config,
                    **kwargs,
                )
                for chunk in stream:
                    yield chunk
                    fallback_output = (
                        chunk if fallback_output is None else fallback_output + chunk
                    )
            run_manager.on_chain_end(fallback_output)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input if isinstance(input, dict) else {"input": input},
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            # Collect all primary chunks first
            child_config = patch_config(config, callbacks=run_manager.get_child())
            chunks: list[AIMessageChunk] = []
            accumulated: AIMessageChunk | None = None
            with set_config_context(child_config):
                astream = self.primary.astream(
                    input,
                    config,
                    logprobs=True,
                    **kwargs,
                )
                async for chunk in astream:
                    chunks.append(chunk)
                    accumulated = chunk if accumulated is None else accumulated + chunk

            accumulated = _require_accumulated(accumulated)
            confidence = self._calculate_confidence(accumulated)
            if confidence >= self.threshold:
                for chunk in chunks:
                    yield chunk
                await run_manager.on_chain_end(accumulated)
                return

            # Fallback: stream from the stronger model
            child_config = patch_config(config, callbacks=run_manager.get_child())
            fallback_output: AIMessageChunk | None = None
            with set_config_context(child_config):
                astream = self.fallback.astream(input, config, **kwargs)
                async for chunk in astream:
                    yield chunk
                    fallback_output = (
                        chunk if fallback_output is None else fallback_output + chunk
                    )
            await run_manager.on_chain_end(fallback_output)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
