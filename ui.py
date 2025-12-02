import base64
import streamlit as st
from datetime import datetime
from scrape import scrape_multiple
from search import get_search_results
from llm_utils import BufferedStreamingHandler, get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary
from config import LOW_RESOURCE_MODE, LOW_RESOURCE_THREADS, LOW_RESOURCE_MAX_ENDPOINTS


# Server-side summary caches
@st.cache_resource
def get_summary_cache_dict():
    # Simple in-process cache for streaming path (fast key lookup)
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_generate_summary(model_name: str, query: str, scraped: dict, prompt_signature: str = "v1") -> str:
    # Build a fresh non-UI LLM instance; output is cached by Streamlit
    llm = get_llm(model_name)
    return generate_summary(llm, query, scraped)


# Cache expensive backend calls
@st.cache_data(ttl=200, show_spinner=False)
def cached_search_results(refined_query: str, threads: int, max_endpoints: int | None):
    return get_search_results(
        refined_query.replace(" ", "+"),
        max_workers=threads,
        max_endpoints=max_endpoints,
    )


@st.cache_data(ttl=200, show_spinner=False)
def cached_scrape_multiple(filtered: list, threads: int):
    return scrape_multiple(filtered, max_workers=threads)


# Streamlit page configuration
st.set_page_config(
    page_title="Robin: AI-Powered Dark Web OSINT Tool",
    page_icon="🕵️‍♂️",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
            .colHeight {
                max-height: 40vh;
                overflow-y: auto;
                text-align: center;
            }
            .pTitle {
                font-weight: bold;
                color: #FF4B4B;
                margin-bottom: 0.5em;
            }
            .aStyle {
                font-size: 18px;
                font-weight: bold;
                padding: 5px;
                padding-left: 0px;
                text-align: center;
            }
    </style>""",
    unsafe_allow_html=True,
)


# Sidebar
st.sidebar.title("Robin")
st.sidebar.text("AI-Powered Dark Web OSINT Tool")
st.sidebar.markdown(
    """Made by [Apurv Singh Gautam](https://www.linkedin.com/in/apurvsinghgautam/)"""
)
st.sidebar.subheader("Settings")
low_bandwidth = st.sidebar.checkbox(
    "Low-bandwidth mode",
    value=LOW_RESOURCE_MODE,
    help="Reduces engines, threads, streaming, and summary size for weak networks/CPUs.",
)
model_options = get_model_choices()
# Prefer lightweight defaults when available
preferred_light = ["gpt-5-nano", "gemini-2.5-flash-lite", "gpt-5-mini"]
def_idx = 0
for pref in preferred_light:
    try:
        def_idx = next(
            idx for idx, name in enumerate(model_options) if name.lower() == pref
        )
        break
    except StopIteration:
        continue
model = st.sidebar.selectbox(
    "Select LLM Model",
    model_options,
    index=def_idx if model_options else 0,
    key="model_select",
)
if any(name not in {"gpt4o", "gpt-4.1", "claude-3-5-sonnet-latest", "llama3.1", "gemini-2.5-flash"} for name in model_options):
    st.sidebar.caption("Locally detected Ollama models are automatically added to this list.")
threads = st.sidebar.slider(
    "Scraping Threads",
    1,
    8 if low_bandwidth else 16,
    LOW_RESOURCE_THREADS if low_bandwidth else 4,
    key="thread_slider",
)
max_endpoints = LOW_RESOURCE_MAX_ENDPOINTS if low_bandwidth else None

# Additional performance toggles
minimal_ui = st.sidebar.checkbox(
    "Minimal UI (hide banners/cards)", value=low_bandwidth, help="Hide logo and step cards to reduce DOM work."
)
plain_text_summary = st.sidebar.checkbox(
    "Plain text summary (faster)", value=low_bandwidth, help="Render summary as plain text (no markdown parsing)."
)
if not low_bandwidth:
    stream_interval_ms = st.sidebar.slider(
        "Streaming update interval (ms)",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Higher values reduce UI update frequency and browser load.",
        key="stream_interval_ms",
    )
else:
    stream_interval_ms = 600


# Main UI - logo and input
if not minimal_ui:
    _, logo_col, _ = st.columns(3)
    with logo_col:
        st.image(".github/assets/robin_logo.png", width=200)

# Display text box and button
with st.form("search_form", clear_on_submit=True):
    col_input, col_button = st.columns([10, 1])
    query = col_input.text_input(
        "Enter Dark Web Search Query",
        placeholder="Enter Dark Web Search Query",
        label_visibility="collapsed",
        key="query_input",
    )
    run_button = col_button.form_submit_button("Run")

# Display a status message
status_slot = st.empty()
# Pre-allocate three placeholders-one per card
cols = st.columns(3)
p1, p2, p3 = [col.empty() for col in cols]
# Summary placeholders
summary_container_placeholder = st.empty()


# Process the query
if run_button and query:
    # Stage 1 - Load LLM
    with status_slot.container():
        with st.spinner("🔄 Loading LLM..."):
            llm = get_llm(model)

    # Stage 2 - Refine query
    with status_slot.container():
        with st.spinner("🔄 Refining query..."):
            refined = refine_query(llm, query)
    if not minimal_ui:
        p1.container(border=True).markdown(
            f"<div class='colHeight'><p class='pTitle'>Refined Query</p><p>{refined}</p></div>",
            unsafe_allow_html=True,
        )

    # Stage 3 - Search dark web
    with status_slot.container():
        with st.spinner("🔍 Searching dark web..."):
            results = cached_search_results(
                refined, threads, max_endpoints
            )
    if not minimal_ui:
        p2.container(border=True).markdown(
            f"<div class='colHeight'><p class='pTitle'>Search Results</p><p>{len(results)}</p></div>",
            unsafe_allow_html=True,
        )

    # Stage 4 - Filter results
    with status_slot.container():
        with st.spinner("🗂️ Filtering results..."):
            filtered = filter_results(
                llm, refined, results
            )
    if not minimal_ui:
        p3.container(border=True).markdown(
            f"<div class='colHeight'><p class='pTitle'>Filtered Results</p><p>{len(filtered)}</p></div>",
            unsafe_allow_html=True,
        )

    # Stage 5 - Scrape content
    with status_slot.container():
        with st.spinner("📜 Scraping content..."):
            scraped = cached_scrape_multiple(
                filtered, threads
            )

    # Stage 6 - Summarize
    # 6a) Prepare local accumulator for streaming text
    summary_text_accum = ""

    # 6c) UI callback for each chunk
    def ui_emit(chunk: str):
        nonlocal summary_text_accum
        summary_text_accum += chunk
        if plain_text_summary:
            summary_slot.text(summary_text_accum)
        else:
            summary_slot.markdown(summary_text_accum)

    with summary_container_placeholder.container():  # border=True, height=450):
        hdr_col, btn_col = st.columns([4, 1], vertical_alignment="center")
        with hdr_col:
            st.subheader(":red[Investigation Summary]", anchor=None, divider="gray")
        summary_slot = st.empty()

    # 6d) Streaming vs final-only based on low-bandwidth mode
    with status_slot.container():
        with st.spinner("✍️ Generating summary..."):
            # Build cache key based on model, query, and scraped content
            import hashlib, json
            cache_key_payload = {
                "model": model,
                "query": query,
                # Sort keys for deterministic hashing
                "scraped": {k: scraped[k] for k in sorted(scraped.keys())},
                "prompt_ver": "v1",
            }
            cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()

            # Try fast in-process cache first
            summary_cache = get_summary_cache_dict()
            cached_text = summary_cache.get(cache_key)

            if low_bandwidth:
                # Disable streaming; prefer cached result or compute via server-side cache
                if cached_text is None:
                    summary_text_accum = cached_generate_summary(model, query, scraped, "v1")
                    summary_cache[cache_key] = summary_text_accum
                else:
                    summary_text_accum = cached_text
                if plain_text_summary:
                    summary_slot.text(summary_text_accum)
                else:
                    summary_slot.markdown(summary_text_accum)
            else:
                if cached_text is not None:
                    # If we have a cache hit, use it directly (no streaming needed)
                    summary_text_accum = cached_text
                    if plain_text_summary:
                        summary_slot.text(summary_text_accum)
                    else:
                        summary_slot.markdown(summary_text_accum)
                else:
                    # Stream first-run for better UX; store result after finish
                    stream_handler = BufferedStreamingHandler(
                        ui_callback=ui_emit,
                        min_emit_interval_seconds=max(0.1, stream_interval_ms / 1000.0),
                    )
                    llm.callbacks = [stream_handler]
                    _ = generate_summary(llm, query, scraped)
                    # After streaming completes, persist to fast cache
                    summary_cache[cache_key] = summary_text_accum

    with btn_col:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"summary_{now}.md"
        b64 = base64.b64encode(summary_text_accum.encode()).decode()
        href = f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" download="{fname}">Download</a></div>'
        st.markdown(href, unsafe_allow_html=True)
    status_slot.success("✔️ Pipeline completed successfully!")
