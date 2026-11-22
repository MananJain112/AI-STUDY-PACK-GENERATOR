import os
import io
import json
import re
import hashlib
from collections import Counter

import numpy as np
import streamlit as st
from pypdf import PdfReader
from pptx import Presentation

from providers import PROVIDERS, build_provider, list_models, friendly_error, ping_model, model_is_dead

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")

st.title("📚 AI-Powered Lecture Notes to Study Pack Generator")
st.caption("Turn your lecture notes into summaries, concepts, definitions, practice questions and answer keys.")


def secret_or_env(name):
    if not name:
        return None
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, None)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _live_models(provider_name, api_key, base_url):
    # Exceptions are not cached, so a failed load is retried on the next click.
    return list_models(provider_name, api_key, base_url)


def live_models(provider_name, api_key, base_url):
    """Live model list, cached 1 hour on success. Listing models costs no tokens."""
    try:
        return _live_models(provider_name, api_key, base_url), None
    except Exception as exc:
        return [], friendly_error(exc, provider_name, "-")


def check_model(provider_name, api_key, model, base_url):
    """Returns (status, message). status: ok | dead | unknown.
    'ok' is remembered for the session, so each model is pinged once.
    'unknown' (bad key, rate limit, network) is not remembered, so it retries."""
    seen = st.session_state.setdefault("model_ok", set())
    k = hashlib.sha256(json.dumps([provider_name, api_key, model, base_url]).encode()).hexdigest()
    if k in seen:
        return "ok", ""
    try:
        ping_model(provider_name, api_key, model, base_url)
        seen.add(k)
        return "ok", ""
    except Exception as exc:
        return ("dead" if model_is_dead(exc) else "unknown"), friendly_error(exc, provider_name, model)


with st.sidebar:
    st.header("🤖 AI Provider")
    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()))
    cfg = PROVIDERS[provider_name]

    base_url = None
    if provider_name == "Custom (OpenAI-compatible)":
        base_url = st.text_input("Base URL", placeholder="https://your-server/v1")
    elif provider_name == "Ollama (local)":
        base_url = st.text_input("Ollama URL", value=cfg["base_url"])

    typed_key = st.text_input(
        "API key" + ("" if cfg["needs_key"] else " (optional)"),
        type="password",
        help="Key typed here is used first. If empty, the app reads "
             + (f"`{cfg['env']}` from env or Streamlit secrets." if cfg["env"] else "no env var."),
    )
    api_key = typed_key.strip() or secret_or_env(cfg["env"])
    if api_key:
        st.caption("✅ Key ready" + (" (typed)" if typed_key.strip() else f" (from {cfg['env']})"))
    elif cfg["needs_key"]:
        st.caption(f"⚠️ No key yet. Type one or set `{cfg['env']}`.")

    # Load the models this key can really use, so retired names never show up.
    models, list_error = [], None
    can_list = api_key or not cfg["needs_key"]
    if can_list and not (provider_name == "Custom (OpenAI-compatible)" and not (base_url or "").strip()):
        models, list_error = live_models(provider_name, api_key, base_url)
    # Models proven broken (retired, no access) this session are dropped from the list.
    dead = st.session_state.setdefault("dead_models", set())
    models = [m for m in models if (provider_name, m) not in dead]
    if list_error:
        st.caption("⚠️ Could not load model list. " + list_error)
    elif not can_list:
        st.caption("Enter your API key to load the models you can use.")

    model_choice = st.selectbox("Model", models + ["Other (type below)"],
                                # New list (key typed, model removed) -> fresh pick from the top.
                                key="model_" + hashlib.md5(json.dumps([provider_name] + models).encode()).hexdigest())
    if model_choice == "Other (type below)":
        model = st.text_input("Model name").strip()
    else:
        model = model_choice

    # Prove the chosen model works with a ~10-token call (once per hour, cached).
    model_ok = False
    if model and can_list:
        with st.spinner(f"Checking {model}..."):
            status, msg = check_model(provider_name, api_key, model, base_url)
        if status == "ok":
            model_ok = True
            st.caption(f"✅ `{model}` works. {len(models)} usable models listed.")
        elif status == "dead" and model_choice != "Other (type below)":
            dead.add((provider_name, model))
            st.toast(f"Removed `{model}`: not usable with this key.")
            st.rerun()  # list redraws without it and picks the next model
        else:
            st.caption("⚠️ " + msg)

    st.divider()
    st.header("⚙️ Study Settings")
    difficulty = st.selectbox("Difficulty level", ["Beginner", "Intermediate", "Advanced"])
    question_count = st.slider("Number of practice questions", 5, 20, 5)
    include_mcq = st.checkbox("Include MCQs", value=True)
    prompt_mode = st.radio("Prompt mode", ["Optimized prompt", "Baseline prompt", "Compare both"])
    run_eval = st.checkbox("AI self-evaluation (+1 API call)", value=False,
                           help="Scores the pack for accuracy, relevance, completeness and source alignment.")
    calls = (2 if prompt_mode == "Compare both" else 1) + (1 if run_eval else 0)
    st.caption(f"💳 API calls per generate: **{calls}**. Reading the file and picking key parts "
               "is free (runs locally). Same file + same settings = no new call.")

# Max characters of notes sent to the model (~2.5k tokens). Keeps input cost small.
MAX_SOURCE_CHARS = 10000


@st.cache_data(show_spinner=False)
def extract_text_cached(name, data):
    """Local parse, cached so widget clicks don't re-read the file."""
    return extract_text(name, io.BytesIO(data))


def extract_text(name, fileobj):
    name = name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(fileobj)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".pptx"):
        prs = Presentation(fileobj)
        parts = []
        for slide_no, slide in enumerate(prs.slides, 1):
            parts.append(f"Slide {slide_no}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
        return "\n".join(parts)
    if name.endswith(".txt"):
        return fileobj.getvalue().decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Please upload PDF, PPTX or TXT.")


def make_chunks(text, chunk_size=1800, overlap=250):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf(counters):
    n = len(counters)
    df = Counter(t for c in counters for t in c)
    idf = {t: np.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
    vecs = []
    for c in counters:
        v = {t: k * idf[t] for t, k in c.items()}
        norm = np.sqrt(sum(x * x for x in v.values())) + 1e-10
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs


def select_source(text, chunk_size=1800, budget=MAX_SOURCE_CHARS):
    """Pick the notes to send. Runs locally, costs no API credit.

    Short notes go in whole. Long notes: rank chunks by how close they are to
    the whole document (TF-IDF cosine), keep the best ones that fit the budget,
    and put them back in note order.
    """
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= budget:
        return [clean], "full notes (short enough to send whole)"
    chunks = make_chunks(clean, chunk_size=chunk_size)
    vecs = _tfidf([Counter(_tokens(c)) for c in chunks] + [Counter(_tokens(clean))])
    doc_vec, chunk_vecs = vecs[-1], vecs[:-1]
    scores = [sum(w * doc_vec.get(t, 0.0) for t, w in v.items()) for v in chunk_vecs]
    keep = max(1, budget // chunk_size)
    top = sorted(int(i) for i in np.argsort(scores)[::-1][:keep])
    return [chunks[i] for i in top], f"top {len(top)} of {len(chunks)} sections (local TF-IDF)"


BASELINE_PROMPT = """
Act as an educational assistant. Read the lecture notes and create a study pack
with a short summary, 5 key concepts, 5 practice questions, and an answer key.
Use simple English, clear headings, and only information supported by the notes.
Return valid JSON with keys: summary, key_concepts, definitions, questions, answer_key.
"""

OPTIMIZED_PROMPT = """
ROLE: You are an accurate educational study-material designer.
TASK: Create an exam-oriented study pack using ONLY the supplied source excerpts.
LEARNER SETTINGS: Difficulty = {difficulty}; number of practice questions = {count};
include MCQs = {mcq}.
RULES:
1. Do not invent facts or use outside knowledge. If information is absent, say so.
2. Preserve important terminology and meaning from the source.
3. Make the summary concise but cover the main ideas.
4. Provide key concepts with clear explanations and a separate list of definitions.
5. Questions must test different ideas and match the requested difficulty.
6. Every answer must be supported by the source excerpts. Do not guess.
7. For MCQs, provide four options and identify the correct option in the answer key.
8. Use clear, student-friendly language and valid JSON only.
OUTPUT JSON SCHEMA:
{{
 "summary": "string",
 "key_concepts": [{{"term":"", "explanation":""}}],
 "definitions": [{{"term":"", "definition":""}}],
 "questions": [{{"question":"", "type":"short/MCQ", "options":[]}}],
 "answer_key": [{{"question_number":1, "answer":"", "source_support":""}}],
 "limitations": ["Mention topics not sufficiently covered in the notes"]
}}
"""

EVAL_RUBRIC = """
Evaluate the study pack against the supplied source notes. Give integer scores from 1 to 5
for accuracy, relevance, completeness and source_alignment. Briefly justify each score.
Do not claim a score is objectively verified; this is an AI-assisted evaluation.
Return JSON: {"accuracy":1,"relevance":1,"completeness":1,"source_alignment":1,
"feedback":"...","improvements":["..."]}
"""


def pack_token_cap(count):
    # Room for summary, concepts, definitions and ~250 tokens per Q+A. Stops runaway output.
    return 1500 + 250 * count


def ask_model(provider, prompt, source_text, count):
    return provider.chat_json(prompt, "SOURCE NOTES / RETRIEVED EXCERPTS:\n" + source_text,
                              temperature=0.2, max_tokens=pack_token_cap(count))


def evaluate_outputs(provider, source_text, output):
    # Compact JSON (no spaces) = fewer input tokens.
    user = ("SOURCE:\n" + source_text +
            "\nSTUDY PACK:\n" + json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return provider.chat_json(EVAL_RUBRIC, user, temperature=0, max_tokens=800)


def run_pipeline(provider, raw_text, difficulty, question_count, include_mcq, prompt_mode, run_eval):
    """Whole generate flow. Returns a results dict for show_results."""
    sections, method = select_source(raw_text)
    grounded_text = "\n\n--- SOURCE EXCERPT ---\n\n".join(sections)
    packs = {}
    if prompt_mode in ["Baseline prompt", "Compare both"]:
        packs["Baseline Prompt"] = ask_model(provider, BASELINE_PROMPT, grounded_text, 5)
    if prompt_mode in ["Optimized prompt", "Compare both"]:
        optimized = OPTIMIZED_PROMPT.format(
            difficulty=difficulty, count=question_count,
            mcq="yes" if include_mcq else "no"
        )
        packs["Optimized Prompt"] = ask_model(provider, optimized, grounded_text, question_count)
    chosen = packs.get("Optimized Prompt") or packs["Baseline Prompt"]

    evaluation, eval_error = None, ""
    if run_eval:
        try:
            evaluation = evaluate_outputs(provider, grounded_text, chosen)
        except Exception as exc:
            eval_error = str(exc)
    return {"retrieval": method, "packs": packs, "chosen": chosen,
            "evaluation": evaluation, "eval_error": eval_error, "ran_eval": run_eval}


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def render_pack(data, label):
    st.subheader(label)
    st.markdown("### 📝 Summary")
    st.write(data.get("summary", ""))
    st.markdown("### 💡 Key Concepts")
    for item in as_list(data.get("key_concepts")):
        if isinstance(item, dict):
            st.markdown(f"**{item.get('term','')}** — {item.get('explanation','')}")
        else:
            st.write(item)
    st.markdown("### 📖 Definitions")
    for item in as_list(data.get("definitions")):
        if isinstance(item, dict):
            st.markdown(f"**{item.get('term','')}**: {item.get('definition','')}")
        else:
            st.write(item)
    st.markdown("### ✍️ Practice Questions")
    for i, item in enumerate(as_list(data.get("questions")), 1):
        if isinstance(item, dict):
            st.markdown(f"**Q{i}. {item.get('question','')}**")
            for option in as_list(item.get("options")):
                st.write(f"- {option}")
        else:
            st.write(f"{i}. {item}")
    st.markdown("### ✅ Answer Key")
    for item in as_list(data.get("answer_key")):
        if isinstance(item, dict):
            st.write(f"Q{item.get('question_number','')}: {item.get('answer','')}")
            if item.get("source_support"):
                st.caption("Source support: " + str(item["source_support"]))
        else:
            st.write(item)
    if data.get("limitations"):
        st.markdown("### ⚠️ Limitations in the Notes")
        for limitation in as_list(data["limitations"]):
            st.write("- " + str(limitation))


def to_markdown(chosen):
    md = f"# AI Study Pack\n\n## Summary\n{chosen.get('summary','')}\n\n## Key Concepts\n"
    for x in as_list(chosen.get("key_concepts")):
        md += f"- **{x.get('term','')}**: {x.get('explanation','')}\n" if isinstance(x, dict) else f"- {x}\n"
    md += "\n## Definitions\n"
    for x in as_list(chosen.get("definitions")):
        md += f"- **{x.get('term','')}**: {x.get('definition','')}\n" if isinstance(x, dict) else f"- {x}\n"
    md += "\n## Questions\n"
    for i, x in enumerate(as_list(chosen.get("questions")), 1):
        if isinstance(x, dict):
            md += f"{i}. {x.get('question','')}\n"
            for opt in as_list(x.get("options")):
                md += f"   - {opt}\n"
        else:
            md += f"{i}. {x}\n"
    md += "\n## Answer Key\n"
    for x in as_list(chosen.get("answer_key")):
        if isinstance(x, dict):
            md += f"- Q{x.get('question_number','')}: {x.get('answer','')}\n"
            if x.get("source_support"):
                md += f"  - Source: {x['source_support']}\n"
        else:
            md += f"- {x}\n"
    if chosen.get("limitations"):
        md += "\n## Limitations\n" + "".join(f"- {x}\n" for x in as_list(chosen["limitations"]))
    return md


def show_results(res):
    st.caption(f"Made with **{res['provider']}** · model `{res['model']}` · source: {res['retrieval']}"
               + (" · ♻️ from cache, no API call" if res.get("from_cache") else ""))
    results = res["packs"]
    if len(results) == 2:
        left, right = st.columns(2)
        with left:
            render_pack(results["Baseline Prompt"], "Baseline Prompt Output")
        with right:
            render_pack(results["Optimized Prompt"], "Optimized Prompt Output")
    else:
        label, pack = next(iter(results.items()))
        render_pack(pack, label + " Output")

    st.divider()
    st.subheader("📊 Prompt Evaluation" + (" (Optimized output)" if len(results) == 2 else ""))
    evaluation = res["evaluation"]
    if not res.get("ran_eval"):
        st.caption("Off. Tick **AI self-evaluation** in the sidebar to score the pack (+1 API call).")
    elif evaluation is None:
        st.warning("Evaluation failed: " + res.get("eval_error", "unknown error"))
    else:
        cols = st.columns(4)
        for col, key, title in zip(
            cols, ["accuracy", "relevance", "completeness", "source_alignment"],
            ["Accuracy", "Relevance", "Completeness", "Source alignment"]
        ):
            col.metric(title, f"{evaluation.get(key, '—')}/5")
        st.write(evaluation.get("feedback", ""))
        for suggestion in as_list(evaluation.get("improvements")):
            st.write("- " + str(suggestion))

    st.download_button("⬇️ Download Study Pack (.md)", to_markdown(res["chosen"]),
                       file_name="study_pack.md")


uploaded = st.file_uploader("Upload your lecture notes", type=["pdf", "pptx", "txt"])
if uploaded:
    try:
        raw_text = extract_text_cached(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    if len(raw_text.strip()) < 40:
        st.error("Very little text was extracted. If this is a scanned PDF, run OCR first or upload a text-based PDF.")
        st.stop()
    st.success(f"Successfully extracted approximately {len(raw_text.split())} words.")
    with st.expander("Preview extracted text"):
        st.text(raw_text[:5000])

    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("cache", {})
    key = hashlib.sha256(json.dumps(
        [text_hash, provider_name, model, base_url, difficulty, question_count,
         include_mcq, prompt_mode, run_eval]).encode("utf-8")).hexdigest()

    if st.button("🚀 Generate Study Pack", type="primary", use_container_width=True):
        problem = None
        if not model:
            problem = "Pick or type a model name."
        elif can_list and not model_ok:
            problem = f"`{model}` did not pass the check. Pick another model (see sidebar)."
        elif cfg["needs_key"] and not api_key:
            problem = f"No API key for {provider_name}. Type it in the sidebar or set `{cfg['env']}`."
        elif provider_name == "Custom (OpenAI-compatible)" and not (base_url or "").strip():
            problem = "Type the base URL for your custom provider."
        if problem:
            st.error(problem)
            st.stop()

        if key in cache:
            cache[key]["from_cache"] = True
        else:
            try:
                provider = build_provider(provider_name, api_key, model, base_url=base_url)
                with st.spinner(f"Generating study material with {provider_name}..."):
                    res = run_pipeline(provider, raw_text, difficulty, question_count,
                                       include_mcq, prompt_mode, run_eval)
                res.update(provider=provider_name, model=model, text_hash=text_hash, from_cache=False)
                cache[key] = res
            except Exception as exc:
                if model_is_dead(exc):
                    st.session_state.setdefault("dead_models", set()).add((provider_name, model))
                st.error(friendly_error(exc, provider_name, model))
                st.stop()
        st.session_state["shown"] = key

    # Results live in session state so they survive reruns (e.g. download click).
    # Only show them if they belong to the file now loaded.
    res = cache.get(st.session_state.get("shown"))
    if res and res["text_hash"] == text_hash:
        show_results(res)
else:
    st.info("Upload a PDF, PPTX or TXT file to begin.")
    st.markdown("**Demo input:** Photosynthesis uses sunlight, carbon dioxide and water to produce glucose and oxygen. Chlorophyll absorbs light energy.")
