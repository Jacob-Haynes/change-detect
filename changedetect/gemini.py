"""Optional cloud 'AI suggestion' per change via the google-genai SDK.

Off unless GEMINI_API_KEY is set. Sends the before and aligned-after crops of a change to
Google, so use demo footage only.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PROMPT = (
    "You are helping a military ISR training exercise compare two photos of the same small area, "
    "taken at different times and already aligned. Image 1 is BEFORE, image 2 is AFTER. "
    "An automatic detector flagged a change near the centre. In ONE line of at most 20 words, say "
    "what most likely changed (e.g. 'vehicle has left', 'new camouflaged shelter', 'ground dug or "
    "disturbed', 'shadow/lighting only'). If it looks like no real change, say so. No preamble."
)


class GeminiAdvisor:
    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.api_key,
                                        http_options=types.HttpOptions(timeout=30_000))
        return self._client

    def describe(self, before_jpg: bytes, after_jpg: bytes, hint: str = "") -> str | None:
        if not self.enabled:
            return None
        from google.genai import types
        try:
            resp = self._get_client().models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=before_jpg, mime_type="image/jpeg"),
                    types.Part.from_bytes(data=after_jpg, mime_type="image/jpeg"),
                    PROMPT + (f" Detector's own guess: {hint}." if hint else ""),
                ],
            )
            text = (resp.text or "").strip().splitlines()
            self.last_error = None
            return text[0][:200] if text else None
        except Exception as e:
            self.last_error = f"Gemini request failed: {e}"
            logger.warning(self.last_error)
            return None

    def status(self) -> dict:
        return {"enabled": self.enabled, "model": self.model if self.enabled else None,
                "last_error": self.last_error}
