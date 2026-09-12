import base64
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class VisionHandler:

    def __init__(
        self,
        model: str = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        base_url: str | None = None,
        max_tokens: int = 512,
        api_key: str = "ollama",
        timeout: float = 60.0,
        auto_start: bool = True,
    ):
        if base_url is None:
            from supergraph.ingest.vision_sidecar import resolve_base_url
            resolved = resolve_base_url(auto_start=auto_start)
            if resolved is None:
                raise RuntimeError(
                    "No vision endpoint available. Either:\n"
                    "  1. pip install 'supergraphdb[vision]' (bundles a local sidecar)\n"
                    "  2. set SUPERGRAPH_VISION_URL to an OpenAI-compatible /v1 URL\n"
                    "  3. run `supergraph vision serve` to start the sidecar manually"
                )
            base_url = resolved
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._timeout = timeout

    @property
    def model(self):
        return self._model

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "In one or two sentences, describe this image. Include any visible text verbatim."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ],
            }],
        }
        try:
            response = self._post("/chat/completions", payload)
        except urllib.error.URLError as e:
            logger.warning("VisionHandler: request failed: %s", e)
            return ""

        choices = response.get("choices") or []
        if not choices:
            logger.warning("VisionHandler: VLM returned no choices for %d bytes", len(image_bytes))
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if not content or not content.strip():
            logger.warning("VisionHandler: VLM returned empty content for %d bytes", len(image_bytes))
            return ""
        return content

    def is_available(self) -> bool:
        try:
            self._get("/models")
            return True
        except Exception as e:
            logger.debug("vision availability check failed: %s", e, exc_info=True)
            return False
