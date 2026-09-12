"""Shared LLM transport for benchmarks and autoresearch.

Single source of truth for all LLM calls in this repo.
Both benchmark runners (locomo, beam, longmemeval) and the autoresearch
run_loop import from here.

Design:
  * Shared per-run concurrency cap (asyncio.Semaphore). Default derived from
    the provider chain: `:free` models -> 20 req/min (matches OpenRouter
    free-tier cap); local/paid -> higher.
  * Per-call retries with exponential backoff on empty/timeout/429.
  * 429 retry-after header honored when litellm surfaces it.
  * Provider fallback: on total failure for a call, try the next provider
    in the chain before giving up for that call.
  * No "no progress, quit round" heuristic. A slow round is not a broken
    run. The runner keeps going until every call has exhausted retries
    for every provider.

Usage (benchmarks):

    from supergraph.llm_runner import LLMRunner, get_shared_runner
    from tools.autoresearch.providers import load_config, resolve_providers

    runner = LLMRunner(resolve_providers(load_config(), model_priority=[...]))
    answers = await runner.call_many(prompts, max_tokens=1000)

Usage (autoresearch):

    from supergraph.llm_runner import LLMRunner
    runner = LLMRunner(resolve_providers(config), timeout_s=800)
    content, model = runner.call_sync_verbose(prompt, temperature=0.7)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 90
DEFAULT_RETRIES = 3
DEFAULT_MAX_TOKENS = 1000
FREE_TIER_RPM = 20
PAID_DEFAULT_RPM = 120
LOCAL_DEFAULT_RPM = 0

_WINDOW_SECONDS = 60.0

# Canonical QA model priority used by get_shared_runner. Deterministic
# non-reasoning model first, cloud paid fallback second. Kept here so
# benches + autoresearch agree on what "the" eval model is.
QA_MODEL_PRIORITY: list[str] = [
    "gemma4:31b-cloud",       # Ollama cloud tag - primary for QA
    "google/gemma-4-31b-it",  # OpenRouter paid fallback
]


@dataclass(slots=True)
class _Concurrency:
    rpm: int
    timestamps: list[float]
    semaphore: asyncio.Semaphore

    def can_fire_now(self) -> float:
        if self.rpm <= 0:
            return 0.0
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
        if len(self.timestamps) < self.rpm:
            return 0.0
        return max(0.0, self.timestamps[0] + _WINDOW_SECONDS - time.monotonic())


def _infer_rpm(providers: list[dict]) -> int:
    has_free = any(":free" in p.get("litellm_model", "") for p in providers)
    if has_free:
        return FREE_TIER_RPM
    has_local = any(
        "localhost" in (p.get("api_base") or "") or "127.0.0.1" in (p.get("api_base") or "")
        for p in providers
    )
    if has_local and len(providers) == 1:
        return LOCAL_DEFAULT_RPM
    return PAID_DEFAULT_RPM


def _parse_retry_after(err: Exception) -> float:
    text = str(err)
    m = re.search(r"retry[- ]after[^0-9]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if "429" in text or "RateLimit" in text or "rate_limit" in text:
        return 5.0
    return 0.0


class LLMRunner:
    """Shared LLM caller for all bench runners and autoresearch."""

    def __init__(
        self,
        providers: list[dict],
        *,
        rpm: int | None = None,
        max_concurrent: int | None = None,
        retries: int = DEFAULT_RETRIES,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not providers:
            raise ValueError("LLMRunner needs at least one provider")
        self._providers = providers
        resolved_rpm = rpm if rpm is not None else _infer_rpm(providers)
        conc_cap = max_concurrent or (resolved_rpm if resolved_rpm > 0 else 16)
        self._c = _Concurrency(
            rpm=resolved_rpm,
            timestamps=[],
            semaphore=asyncio.Semaphore(conc_cap),
        )
        self._retries = max(1, retries)
        self._timeout_s = max(5, timeout_s)
        self.last_model: str = ""

    @property
    def rpm(self) -> int:
        return self._c.rpm

    async def _wait_rate(self) -> None:
        while True:
            wait = self._c.can_fire_now()
            if wait <= 0:
                self._c.timestamps.append(time.monotonic())
                return
            await asyncio.sleep(min(wait, 1.0))

    async def _litellm_call(
        self,
        prompt: str,
        provider: dict,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import litellm
        litellm.suppress_debug_info = True

        def _sync() -> str:
            resp = litellm.completion(
                model=provider["litellm_model"],
                messages=[{"role": "user", "content": prompt}],
                api_base=provider["api_base"],
                api_key=provider["api_key"],
                stream=False,
                timeout=self._timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync),
            timeout=self._timeout_s + 10,
        )

    async def call_one(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Execute a single prompt. Returns empty string on total failure."""
        async with self._c.semaphore:
            for attempt in range(self._retries):
                for provider in self._providers:
                    await self._wait_rate()
                    try:
                        result = await self._litellm_call(prompt, provider, max_tokens, temperature)
                        if result:
                            self.last_model = provider.get("litellm_model", "")
                            return result
                    except asyncio.TimeoutError:
                        _log.debug("timeout on %s attempt=%d", provider.get("pid"), attempt)
                        continue
                    except Exception as err:
                        wait = _parse_retry_after(err)
                        if wait > 0:
                            _log.info("rate limit on %s, sleeping %.1fs", provider.get("pid"), wait)
                            await asyncio.sleep(wait)
                        else:
                            _log.debug("call failed on %s attempt=%d: %s", provider.get("pid"), attempt, err)
                await asyncio.sleep(2 ** attempt)
            return ""

    async def call_many(
        self,
        prompts: list[str],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        on_progress=None,
    ) -> list[str]:
        """Run N prompts concurrently, respecting rpm + concurrency cap."""
        total = len(prompts)
        results: list[str] = [""] * total
        done = 0
        lock = asyncio.Lock()

        async def _worker(i: int, p: str) -> None:
            nonlocal done
            r = await self.call_one(p, max_tokens=max_tokens, temperature=temperature)
            async with lock:
                results[i] = r
                done += 1
                if on_progress:
                    on_progress(done, total)

        await asyncio.gather(*[_worker(i, p) for i, p in enumerate(prompts)])
        return results

    def call_sync(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Sync wrapper. Safe only from non-async contexts."""
        return asyncio.run(self.call_one(prompt, max_tokens=max_tokens, temperature=temperature))

    def call_sync_verbose(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> tuple[str, str]:
        """Sync call returning (content, model_used). For autoresearch logging."""
        content = self.call_sync(prompt, max_tokens=max_tokens, temperature=temperature)
        return content, self.last_model

    def complete_messages(
        self,
        messages: list[dict],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Sync chat-shaped completion with provider fallback.

        Tries each provider in order across `retries` rounds; returns the first
        non-empty content. Returns "" on total failure. Unlike call_one this
        accepts a full messages list (system + user) and runs synchronously -
        the rpm cap is not enforced here (interactive ingestion is per-message,
        not bulk). Use call_many for rpm-bounded batch throughput.
        """
        import litellm
        litellm.suppress_debug_info = True
        for attempt in range(self._retries):
            for provider in self._providers:
                extra = {"account_id": provider["account_id"]} if provider.get("account_id") else {}
                try:
                    resp = litellm.completion(
                        model=provider["litellm_model"],
                        messages=messages,
                        api_base=provider.get("api_base"),
                        api_key=provider["api_key"],
                        stream=False,
                        timeout=self._timeout_s,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **extra,
                    )
                    content = resp.choices[0].message.content or ""
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    if content:
                        self.last_model = provider["litellm_model"]
                        return content
                except Exception as err:
                    wait = _parse_retry_after(err)
                    if wait > 0:
                        time.sleep(wait)
                    _log.debug("complete_messages failed on %s: %s", provider.get("pid"), err)
            time.sleep(2 ** attempt)
        return ""

    def stream_messages(
        self,
        messages: list[dict],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ):
        """Sync streaming completion. Yields content deltas.

        Provider fallback only happens before the first chunk; once a provider
        starts streaming we commit to it (no cross-provider mid-stream resume).
        Raises RuntimeError if every provider fails before yielding a token.
        """
        import litellm
        litellm.suppress_debug_info = True
        last_err: Exception | None = None
        for provider in self._providers:
            extra = {"account_id": provider["account_id"]} if provider.get("account_id") else {}
            try:
                stream = litellm.completion(
                    model=provider["litellm_model"],
                    messages=messages,
                    api_base=provider.get("api_base"),
                    api_key=provider["api_key"],
                    stream=True,
                    timeout=self._timeout_s,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                self.last_model = provider["litellm_model"]
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
                return
            except Exception as err:
                last_err = err
                _log.debug("stream_messages failed on %s: %s", provider.get("pid"), err)
                continue
        raise RuntimeError(f"all providers failed to stream: {last_err}")


_SHARED: "LLMRunner | None" = None


def get_shared_runner() -> "LLMRunner":
    """Return (or construct) the process-wide LLMRunner.

    Provider chain comes from tools/autoresearch/config.json via
    providers.resolve_providers. Call reset_shared_runner() after
    config changes.
    """
    global _SHARED
    if _SHARED is None:
        from tools.autoresearch.providers import load_config, resolve_providers
        providers = resolve_providers(load_config(), model_priority=QA_MODEL_PRIORITY)
        if not providers:
            raise RuntimeError(
                "No LLM providers resolved. Check tools/autoresearch/config.json "
                "active_provider + provider_fallback_order."
            )
        _SHARED = LLMRunner(providers)
    return _SHARED


def reset_shared_runner() -> None:
    """Drop the cached shared runner so the next call rebuilds it."""
    global _SHARED
    _SHARED = None
