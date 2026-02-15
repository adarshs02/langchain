"""Tests for TieredSemanticRouter."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.tiered_semantic_router import TieredSemanticRouter

# ---------------------------------------------------------------------------
# Fake model fixture
# ---------------------------------------------------------------------------


class FakeModelWithLogprobs(BaseChatModel):
    """Fake chat model that returns configurable logprobs."""

    response_text: str = "Hello world"
    token_logprobs: list[float] = [-0.1, -0.2]  # noqa: RUF012

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-logprobs"

    @override
    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        logprobs_kwarg = kwargs.get("logprobs", False)
        generation_info: dict[str, Any] = {}
        if logprobs_kwarg:
            generation_info["logprobs"] = {
                "content": [
                    {"token": tok, "logprob": lp}
                    for tok, lp in zip(
                        self.response_text.split(), self.token_logprobs, strict=False
                    )
                ]
            }
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=self.response_text),
                    generation_info=generation_info or None,
                )
            ]
        )

    @override
    def _stream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        logprobs_kwarg = kwargs.get("logprobs", False)
        tokens = self.response_text.split()
        for i, token in enumerate(tokens):
            generation_info: dict[str, Any] = {}
            if logprobs_kwarg and i < len(self.token_logprobs):
                generation_info["logprobs"] = {
                    "content": [{"token": token, "logprob": self.token_logprobs[i]}]
                }
            content = token if i == 0 else f" {token}"
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=content),
                generation_info=generation_info or None,
            )

    @override
    async def _astream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk


# ---------------------------------------------------------------------------
# _calculate_confidence tests
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    def test_valid(self) -> None:
        msg = AIMessage(
            content="hi",
            response_metadata={
                "logprobs": {
                    "content": [
                        {"token": "hi", "logprob": -0.1},
                        {"token": "there", "logprob": -0.2},
                    ]
                }
            },
        )
        assert TieredSemanticRouter._calculate_confidence(msg) == pytest.approx(-0.15)

    def test_no_logprobs_raises(self) -> None:
        msg = AIMessage(content="hi")
        with pytest.raises(ValueError, match="does not contain logprobs"):
            TieredSemanticRouter._calculate_confidence(msg)

    def test_empty_content_raises(self) -> None:
        msg = AIMessage(
            content="hi",
            response_metadata={"logprobs": {"content": []}},
        )
        with pytest.raises(ValueError, match="empty logprobs content"):
            TieredSemanticRouter._calculate_confidence(msg)

    def test_inf_logprobs_filtered(self) -> None:
        """Infinite logprob values are excluded from the mean."""
        msg = AIMessage(
            content="hi",
            response_metadata={
                "logprobs": {
                    "content": [
                        {"token": "hi", "logprob": -0.5},
                        {"token": "there", "logprob": float("inf")},
                    ]
                }
            },
        )
        assert TieredSemanticRouter._calculate_confidence(msg) == pytest.approx(-0.5)

    def test_all_nan_raises(self) -> None:
        """All non-finite logprobs should raise ValueError."""
        msg = AIMessage(
            content="hi",
            response_metadata={
                "logprobs": {"content": [{"token": "hi", "logprob": float("nan")}]}
            },
        )
        with pytest.raises(ValueError, match="No finite logprob"):
            TieredSemanticRouter._calculate_confidence(msg)


# ---------------------------------------------------------------------------
# invoke / ainvoke tests
# ---------------------------------------------------------------------------


def _make_router(
    primary_logprobs: list[float],
    fallback_text: str = "fallback response",
    threshold: float = -0.5,
    primary_text: str = "Hello world",
) -> TieredSemanticRouter:
    return TieredSemanticRouter(
        primary=FakeModelWithLogprobs(
            response_text=primary_text,
            token_logprobs=primary_logprobs,
        ),
        fallback=FakeModelWithLogprobs(
            response_text=fallback_text,
            token_logprobs=[-0.01, -0.02],
        ),
        threshold=threshold,
    )


class TestInvoke:
    def test_primary_confident(self) -> None:
        router = _make_router(primary_logprobs=[-0.1, -0.2])
        result = router.invoke("test")
        assert result.content == "Hello world"

    def test_fallback_triggered(self) -> None:
        router = _make_router(primary_logprobs=[-2.0, -3.0])
        result = router.invoke("test")
        assert result.content == "fallback response"

    def test_threshold_boundary(self) -> None:
        """Exact threshold match should return primary (>= comparison)."""
        router = _make_router(primary_logprobs=[-0.5, -0.5], threshold=-0.5)
        result = router.invoke("test")
        assert result.content == "Hello world"

    def test_logprobs_kwarg_injected(self) -> None:
        """Primary model should receive logprobs=True."""
        calls: list[dict[str, Any]] = []

        class SpyModel(FakeModelWithLogprobs):
            @override
            def _generate(
                self,
                messages: list[Any],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                calls.append(kwargs)
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        router = TieredSemanticRouter(
            primary=SpyModel(response_text="ok", token_logprobs=[-0.1]),
            fallback=FakeModelWithLogprobs(response_text="fb", token_logprobs=[-0.01]),
            threshold=-1.0,
        )
        router.invoke("test")
        assert calls[0].get("logprobs") is True

    def test_primary_error_propagates(self) -> None:
        """Exceptions from the primary model propagate correctly."""

        class FailingModel(FakeModelWithLogprobs):
            @override
            def _generate(
                self,
                messages: list[Any],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                msg = "model failure"
                raise RuntimeError(msg)

        router = TieredSemanticRouter(
            primary=FailingModel(),
            fallback=FakeModelWithLogprobs(),
            threshold=-1.0,
        )
        with pytest.raises(RuntimeError, match="model failure"):
            router.invoke("test")


class TestAInvoke:
    @pytest.mark.asyncio
    async def test_primary_confident(self) -> None:
        router = _make_router(primary_logprobs=[-0.1, -0.2])
        result = await router.ainvoke("test")
        assert result.content == "Hello world"

    @pytest.mark.asyncio
    async def test_fallback_triggered(self) -> None:
        router = _make_router(primary_logprobs=[-2.0, -3.0])
        result = await router.ainvoke("test")
        assert result.content == "fallback response"


# ---------------------------------------------------------------------------
# stream / astream tests
# ---------------------------------------------------------------------------


class TestStream:
    def test_primary_confident(self) -> None:
        router = _make_router(primary_logprobs=[-0.1, -0.2])
        chunks = list(router.stream("test"))
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert full == "Hello world"

    def test_fallback_triggered(self) -> None:
        router = _make_router(primary_logprobs=[-2.0, -3.0])
        chunks = list(router.stream("test"))
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert full == "fallback response"


class TestAStream:
    @pytest.mark.asyncio
    async def test_primary_confident(self) -> None:
        router = _make_router(primary_logprobs=[-0.1, -0.2])
        chunks = [chunk async for chunk in router.astream("test")]
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert full == "Hello world"

    @pytest.mark.asyncio
    async def test_fallback_triggered(self) -> None:
        router = _make_router(primary_logprobs=[-2.0, -3.0])
        chunks = [chunk async for chunk in router.astream("test")]
        assert len(chunks) > 0
        full = "".join(c.content for c in chunks)
        assert full == "fallback response"


# ---------------------------------------------------------------------------
# LCEL chain compatibility
# ---------------------------------------------------------------------------


class TestLCELCompatibility:
    def test_chain_compatible(self) -> None:
        """TieredSemanticRouter works in prompt | router | parser chain."""
        router = _make_router(primary_logprobs=[-0.1, -0.2])
        prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
        chain = prompt | router | StrOutputParser()
        result = chain.invoke({"question": "hello"})
        assert isinstance(result, str)
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_is_lc_serializable(self) -> None:
        assert TieredSemanticRouter.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        assert TieredSemanticRouter.get_lc_namespace() == [
            "langchain",
            "schema",
            "runnable",
        ]
