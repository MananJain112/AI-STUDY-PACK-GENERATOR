"""LLM provider layer. One small interface, many back ends.

Every provider exposes:
    chat_json(system, user, temperature, max_tokens) -> dict

Retrieval runs locally (TF-IDF in app.py), so no embedding calls are made.
"""
import json
import re

# name -> settings. "kind" picks the client class.
PROVIDERS = {
    "OpenAI": {
        "kind": "openai",
        "base_url": None,
        "env": "OPENAI_API_KEY",
        "models": [],  # loaded live
        "needs_key": True,
    },
    "Anthropic (Claude)": {
        "kind": "anthropic",
        "env": "ANTHROPIC_API_KEY",
        "models": [],  # loaded live
        "needs_key": True,
    },
    "Google Gemini": {
        "kind": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env": "GEMINI_API_KEY",
        "models": [],  # loaded live from Google
        "chat_extra": {"reasoning_effort": "low"},  # less thinking = fewer billed tokens
        "think_headroom": 4000,
        "needs_key": True,
    },
    "Groq": {
        "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY",
        "models": [],  # loaded live
        "needs_key": True,
    },
    "OpenRouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "env": "OPENROUTER_API_KEY",
        "models": [],  # loaded live
        "needs_key": True,
    },
    "Ollama (local)": {
        "kind": "openai",
        "base_url": "http://localhost:11434/v1",
        "env": None,
        "models": [],  # loaded live from your Ollama
        "needs_key": False,
    },
    "Custom (OpenAI-compatible)": {
        "kind": "openai",
        "base_url": "",
        "env": "CUSTOM_LLM_API_KEY",
        "models": [],
        "needs_key": False,
    },
}

# Model ids that are not chat models. Hidden from the live list.
_NOT_CHAT = re.compile(
    r"embed|tts|whisper|audio|realtime|transcri|image|imagen|veo|dall-e|moderation|"
    r"guard|aqa|live|search|computer-use|robotics|davinci|babbage", re.I)

# Words that mark a cheap, fast default per provider (first match wins the default slot).
_PREFER = {"OpenAI": "mini", "Google Gemini": "flash", "Anthropic (Claude)": "sonnet",
           "Groq": "versatile"}


def list_models(name, api_key, base_url=None):
    """Live chat-model list from the provider. Free call (no tokens). Raises on failure."""
    cfg = PROVIDERS[name]
    if cfg["kind"] == "anthropic":
        import anthropic
        ids = [m.id for m in anthropic.Anthropic(api_key=api_key).models.list(limit=100)]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or "not-needed",
                        base_url=(base_url if base_url is not None else cfg.get("base_url")) or None)
        ids = [m.id.removeprefix("models/") for m in client.models.list()]
    ids = [i for i in ids if not _NOT_CHAT.search(i)]
    if name == "Google Gemini":
        ids = [i for i in ids if i.startswith("gemini")]
    if name == "OpenAI":
        ids = [i for i in ids if i.startswith(("gpt", "o1", "o3", "o4", "o5", "chatgpt"))]
    ids = sorted(set(ids), reverse=True)  # newer versions tend to sort first
    prefer = _PREFER.get(name)

    def rank(i):
        unstable = any(w in i for w in ("preview", "exp", "-00", "latest"))
        return (unstable, not (prefer and prefer in i))
    return sorted(ids, key=rank)


def ping_model(name, api_key, model, base_url=None):
    """Tiny real call to prove a model works. About 10 tokens. Raises on failure."""
    cfg = PROVIDERS[name]
    if cfg["kind"] == "anthropic":
        import anthropic
        anthropic.Anthropic(api_key=api_key).messages.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": "hi"}])
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or "not-needed",
                        base_url=(base_url if base_url is not None else cfg.get("base_url")) or None)
        client.chat.completions.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": "hi"}])


_DEAD_WORDS = ("not found", "deprecated", "no longer", "not supported", "does not exist",
               "decommissioned", "retired", "unsupported model", "invalid model")


def model_is_dead(exc):
    """True when the error means 'this model cannot be used', not 'bad key' or 'busy'."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status in (404, 410):
        return True
    if status in (400, 403) and any(w in text for w in _DEAD_WORDS):
        return True
    return False


def friendly_error(exc, provider_name, model):
    """Turn SDK errors into a plain message with the provider's own reason."""
    status = getattr(exc, "status_code", None)
    detail = getattr(exc, "message", None) or str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            detail = err["message"]
    hint = {
        401: "The API key was rejected. Check it is right and for this provider.",
        403: "The key has no access to this model or API. Check your account or plan.",
        404: f"Model `{model}` was not found. It may be retired or misspelled. Pick another model.",
        429: "Rate limit hit or no credits left. Wait a bit or check your billing.",
    }.get(status)
    if status and status >= 500:
        hint = "The provider had a server problem. Try again in a minute."
    head = f"{provider_name} call failed" + (f" (HTTP {status})" if status else "")
    return f"{head}: {hint + ' ' if hint else ''}Details: {detail}"


# Claude models that take server-side refusal fallbacks.
_CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5",
                           "claude-fable-5-2", "claude-fable-5-1"}


def parse_json(text):
    """Pull a JSON object out of model text. Handles ```json fences and chatter."""
    if text is None:
        raise ValueError("Model returned an empty response.")
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Model did not return valid JSON. Try again or pick another model.")


class OpenAICompatProvider:
    def __init__(self, api_key, model, base_url=None, chat_extra=None, think_headroom=0):
        from openai import OpenAI
        # Local servers (Ollama) need some non-empty key string.
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
        self.model = model
        self.chat_extra = chat_extra or {}
        self.think_headroom = think_headroom

    def chat_json(self, system, user, temperature=0.2, max_tokens=4000):
        from openai import BadRequestError
        kwargs = dict(
            model=self.model,
            temperature=temperature,
            # Hard cap on paid output. Thinking models count thinking here too, so add headroom.
            max_tokens=max_tokens + self.think_headroom,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        try:
            resp = self.client.chat.completions.create(
                response_format={"type": "json_object"}, **self.chat_extra, **kwargs)
        except BadRequestError:
            # Some models reject JSON mode or the extra settings. Retry plain once;
            # a rejected (400) request is not billed.
            resp = self.client.chat.completions.create(**kwargs)
        if resp.choices[0].finish_reason == "length":
            raise ValueError("Answer hit the output limit and was cut off. Ask for fewer questions.")
        return parse_json(resp.choices[0].message.content)


class AnthropicProvider:
    def __init__(self, api_key, model):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat_json(self, system, user, temperature=0.2, max_tokens=4000):
        # Claude 5.x rejects temperature, so it is not sent.
        system = system + "\nRespond with one JSON object only. No markdown fences, no extra text."
        kwargs = dict(
            model=self.model,
            # Thinking tokens count toward max_tokens, so leave headroom above the JSON size.
            max_tokens=max_tokens + 4000,
            # Low effort = little thinking = fewer billed tokens. Fine for a grounded summary task.
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        import anthropic
        if self.model in _CLAUDE_FALLBACK_MODELS:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            try:
                resp = self.client.messages.create(**kwargs)
            except anthropic.BadRequestError:
                # Older Claude models (typed in by hand) may not take effort. Retry without it.
                kwargs.pop("output_config")
                kwargs["max_tokens"] = max_tokens
                resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise ValueError("Claude declined this request. Try different notes or another model.")
        if resp.stop_reason == "max_tokens":
            raise ValueError("Answer hit the output limit and was cut off. Ask for fewer questions.")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_json(text)


def build_provider(name, api_key, model, base_url=None):
    cfg = PROVIDERS[name]
    if cfg["kind"] == "anthropic":
        return AnthropicProvider(api_key, model)
    return OpenAICompatProvider(
        api_key, model,
        base_url=base_url if base_url is not None else cfg.get("base_url"),
        chat_extra=cfg.get("chat_extra"),
        think_headroom=cfg.get("think_headroom", 0),
    )
