from __future__ import annotations
import os
import io
import json
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# --- Windows/Streamlit threading safety ---
import nest_asyncio
nest_asyncio.apply()

import streamlit as st

# LangChain core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM providers
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Chains / docs / prompts
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Reranking (optional)
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except Exception:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None  # type: ignore

# Concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# Streamlit Page Config FIRST
# =============================
st.set_page_config(page_title="Self-Improving RAG — Groq/OpenAI (Windows-safe)", layout="wide")
st.caption("Uploads PDFs → auto-tunes RAG → evaluates → chat with the best config.")

# =============================
# Styled CSS
# =============================
st.markdown(
    """
    <style>
    body { background-color: #f4f6fb; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.0rem !important; padding-bottom: 2rem !important; max-width: 1200px; }
    h1, h2, h3, h4 { color: #2c3e50; font-weight: 700; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #eef4ff 0%, #f9fbff 100%); border-right: 1px solid #dce3f0; }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: #4455a5; }
    section[data-testid="stFileUploader"] > div { border: 2px dashed #b0c4ff !important; background: #f8fbff !important; border-radius: 12px !important; padding: 0.2rem 0.8rem !important; }
    [data-testid="stFileUploader"] > div:first-child { height: 120px !important; padding: 8px !important; }
    section[data-testid="stFileUploader"] button { background: linear-gradient(90deg, #5A60F2, #5EC5FF); color: white !important; border-radius: 8px !important; font-size: 0.9rem !important; padding: 0.3rem 0.9rem !important; border: none !important; transition: 0.3s ease; }
    section[data-testid="stFileUploader"] button:hover { box-shadow: 0px 0px 8px rgba(90, 96, 242, 0.4); transform: scale(1.02); }
    .stButton>button { background: linear-gradient(90deg,#6A67F2,#5EC5FF); color:white !important; border-radius: 12px !important; padding: 0.55rem 1.2rem !important; font-weight: 600 !important; border: none; transition: 0.25s ease-in-out; box-shadow: 0px 2px 6px rgba(0,0,0,0.1); }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 4px 10px rgba(90, 96, 242, 0.4); }
    .stChatMessage { border-radius: 16px !important; padding: 1rem 1.2rem !important; margin-bottom: 0.8rem !important; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .stChatMessage.user { background: #e8f0ff !important; border-left: 4px solid #5A60F2 !important; }
    .stChatMessage.assistant { background: #ffffff !important; border-left: 4px solid #5EC5FF !important; }
    .stExpander { border-radius: 12px !important; background: #ffffff !important; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-top: 0.5rem !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #5A60F2, #5EC5FF) !important; }
    hr { border: none; height: 1px; background: #e4e7ef; margin: 1rem 0; }
    .stAlert { border-radius: 10px !important; }
    .stMarkdown p { margin-bottom: 0.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Sidebar — provider & settings
# =============================
with st.sidebar:
    st.header("⚙️ Settings")

    # Provider
    with st.expander("🤖 Model Provider", expanded=True):
        provider = st.selectbox("Model Provider", ["Groq", "OpenAI"], index=0, key="provider_select")
        if provider == "OpenAI":
            st.session_state.openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
            st.session_state.groq_key = None
        else:
            st.session_state.groq_key = st.text_input("Groq API Key", type="password", key="groq_key_input")
            st.session_state.openai_key = None

    st.divider()

    with st.expander("📎 Embeddings & Search", expanded=False):
        embed_model_name = st.selectbox(
            "Embedding Model",
            ["intfloat/e5-small-v2", "BAAI/bge-small-en-v1.5", "thenlper/gte-small"],
            index=1,
            key="embed_model_name_select",
        )
        st.caption("Tip: e5-small or BGE-small = fast + strong")

    st.divider()

    with st.expander("🔬 Auto-Tune RAG", expanded=False):
        chunk_sizes = st.multiselect("Chunk sizes", [256, 384, 512, 768], default=[384, 512], key="chunk_sizes")
        overlaps = st.multiselect("Overlaps", [20, 40, 60, 80], default=[40], key="overlaps")
        topk_values = st.multiselect("Top-k", [3, 5, 7, 10], default=[5, 7], key="topk_values")
        use_rerank = st.multiselect("Rerank?", ["off", "on"], default=["off", "on"], key="use_rerank")
        n_questions = st.slider("Synthetic Qs", 3, 12, 5, key="n_questions")
        st.caption("More Qs = slower but better tuning")

    st.divider()

    with st.expander("🕸️ Graph DB (Optional)", expanded=False):
        use_graph = st.checkbox("Enable Neo4j Graph-RAG", value=False, key="use_graph")
        neo4j_uri = st.text_input("NEO4J_URI", value="bolt://localhost:7687", key="neo4j_uri")
        neo4j_user = st.text_input("NEO4J_USER", value="neo4j", key="neo4j_user")
        neo4j_pwd = st.text_input("NEO4J_PASSWORD", type="password", key="neo4j_pwd")
        st.caption("Run Neo4j locally or cloud")

# =============================
# Helper: LLM factory & utils
# =============================

def get_llm(provider: str, openai_key: Optional[str], groq_key: Optional[str], temperature: float = 0.0, model_name: Optional[str] = None):
    if provider == "OpenAI":
        key = (openai_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            st.error("OpenAI key required for OpenAI provider.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = key
        return ChatOpenAI(temperature=temperature, model=model_name or "gpt-4o-mini")
    else:
        key = (groq_key or os.getenv("GROQ_API_KEY", "")).strip()
        if not key:
            st.error("Groq key required for Groq provider.")
            st.stop()
        os.environ["GROQ_API_KEY"] = key
        return ChatGroq(temperature=temperature, model_name=model_name or "llama-3.1-8b-instant")


def to_text(resp) -> str:
    """LangChain .invoke may return BaseMessage or str; normalize to plain string."""
    try:
        # BaseMessage-like
        content = getattr(resp, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass
    # Fallback
    return str(resp)


# =============================
# Upload & Ingest
# =============================
col1, col2 = st.columns(2)

all_docs: List[Document] = []

with col1:
    st.subheader("📄 Upload PDFs")
    files = st.file_uploader("Drop one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            with open(tmp_path, "wb") as out:
                out.write(f.read())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
        st.success(f"Loaded {len(all_docs)} pages from {len(files)} PDF(s).")
    else:
        st.info("Upload PDFs to continue.")

# =============================
# Graph ingest (optional, preview only)
# =============================
if st.session_state.get("use_graph") and files:
    try:
        from langchain_community.graphs import Neo4jGraph
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        st.error("Install graph extras: pip install langchain-community neo4j")
        st.stop()

    st.markdown("#### 🕸️ Build Knowledge Graph (triples)")
    build_graph = st.button("Extract relations & upsert into Neo4j", disabled=not bool(files))

    if build_graph:
        with st.spinner("Connecting to Neo4j and extracting triples…"):
            try:
                graph = Neo4jGraph(url=st.session_state["neo4j_uri"], username=st.session_state["neo4j_user"], password=st.session_state["neo4j_pwd"])
            except Exception as e:
                st.error(f"Neo4j connection failed: {e}")
                st.stop()

            llm_ie = get_llm(provider, st.session_state.get("openai_key"), st.session_state.get("groq_key"), temperature=0.0)
            tpl = ChatPromptTemplate.from_messages([
                ("system", "Extract factual triples (head, relation, tail) from the text. Return JSON list of objects with keys: head, relation, tail. Be concise."),
                ("human", "Text:{chunk}")
            ])

            preview_texts = [d.page_content[:1500] for d in all_docs[:6]]
            imported = 0
            for raw in preview_texts:
                prompt_msgs = tpl.format_messages(chunk=raw)
                resp = llm_ie.invoke(prompt_msgs)
                txt = to_text(resp)
                triples: List[Dict[str, str]] = []
                try:
                    triples = json.loads(txt)
                except Exception:
                    # fallback: try simple pipe-delimited extraction
                    lines = [x.strip() for x in txt.splitlines() if x.strip()]
                    for ln in lines:
                        if "|" in ln:
                            parts = [p.strip() for p in ln.split("|")]
                            if len(parts) >= 3:
                                triples.append({"head": parts[0], "relation": parts[1], "tail": parts[2]})
                for t in triples:
                    h, r, ta = t.get("head"), t.get("relation"), t.get("tail")
                    if not (h and r and ta):
                        continue
                    cypher = (
                        "MERGE (h:Entity {name:$h})"
                        "MERGE (t:Entity {name:$t})"
                        "MERGE (h)-[:REL {type:$r}]->(t)"
                    )
                    try:
                        graph.query(cypher, params={"h": h[:256], "r": r[:128], "t": ta[:256]})
                        imported += 1
                    except Exception:
                        pass
            st.success(f"Inserted/merged ~{imported} triples into Neo4j.")

# =============================
# RAG Config & Helpers
# =============================
@dataclass
class RagConfig:
    chunk_size: int
    overlap: int
    top_k: int
    rerank: bool
    prompt_style: str  # 'tight' or 'verbose'


def build_splitter(cfg: RagConfig):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ".", " "]
    )


@st.cache_resource(show_spinner=False)
def cached_embedder(name: str):
    return HuggingFaceEmbeddings(model_name=name)


def embedder(name: str):
    return cached_embedder(name)


# Optional reranker
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce: Optional[CrossEncoder] = None
if HAS_CROSS_ENCODER:
    try:
        _ce = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        _ce = None


def rerank_if_enabled(query: str, docs: List[Document], enabled: bool) -> List[Document]:
    if not enabled or _ce is None or not docs:
        return docs
    pairs = [[query, d.page_content] for d in docs]
    scores = _ce.predict(pairs)
    rescored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in rescored]

# Prompts
PROMPT_TIGHT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are a precise assistant. Use ONLY the context below to answer. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

PROMPT_VERBOSE = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "Answer the user's question using the context. Provide citations like [C1], [C2] mapping to chunk numbers. "
        "If unsure, clearly say so.\n\nContext:\n{context}\n\nQuestion: {question}\nDetailed Answer:"
    ),
)


def prompt_for_style(style: str) -> PromptTemplate:
    return PROMPT_TIGHT if style == "tight" else PROMPT_VERBOSE

# Eval prompts
GEN_Q_PROMPT = PromptTemplate(
    input_variables=["context", "n"],
    template=(
        "Create {n} diverse, factual questions that can be answered from the context. "
        "Avoid yes/no. Output as a JSON list of strings only.\n\nContext:\n{context}"
    ),
)

GRADE_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template=(
        "You are a strict evaluator. Given a question, an answer, and the retrieved context, rate faithfulness. "
        "Score 0 to 1 where 1 means the answer is fully supported by the context, 0 means unsupported. "
        "Return ONLY a float number.\n\nQuestion: {question}\nAnswer: {answer}\nContext:\n{context}"
    ),
)


def generate_questions(llm, sample_docs: List[Document], n: int) -> List[str]:
    joined = " ".join(d.page_content[:1500] for d in sample_docs[:5])
    prompt = GEN_Q_PROMPT.format(context=joined, n=n)
    resp = llm.invoke(prompt)
    txt = to_text(resp)
    try:
        data = json.loads(txt)
        return [q for q in data if isinstance(q, str)][:n]
    except Exception:
        # Fallback: naive split
        out: List[str] = []
        for line in txt.splitlines():
            if not line.strip():
                continue
            # remove bullets
            out.append(line.strip("-•* "))
        return out[:n]


def evaluate_config(llm, cfg: RagConfig, raw_docs: List[Document], embed_model_name: str, n_questions: int) -> Tuple[float, Dict]:
    splitter = build_splitter(cfg)
    chunks = splitter.split_documents(raw_docs)

    embeddings = embedder(embed_model_name)
    vectordb = FAISS.from_documents(chunks, embeddings)

    sample_docs = chunks[:6] if len(chunks) > 6 else chunks
    if not sample_docs:
        return 0.0, {"questions": [], "runs": []}

    questions = generate_questions(llm, sample_docs, n_questions)

    total = 0.0
    details = []
    for q in questions:
        retrieved = vectordb.similarity_search(q, k=cfg.top_k)
        retrieved = rerank_if_enabled(q, retrieved, cfg.rerank)
        ctx = " ".join(f"[C{i+1}] " + d.page_content for i, d in enumerate(retrieved))
        prompt = prompt_for_style(cfg.prompt_style).format(question=q, context=ctx)
        ans = to_text(llm.invoke(prompt))
        grade_txt = to_text(llm.invoke(GRADE_PROMPT.format(question=q, answer=ans, context=ctx))).strip()
        try:
            score = float(grade_txt)
        except Exception:
            score = 0.0
        total += max(0.0, min(1.0, score))
        details.append({"q": q, "answer": ans, "score": score})

    avg = total / max(1, len(questions))
    return avg, {"questions": questions, "runs": details}

# =============================
# Auto-Tuning (THREADS; Windows-safe)
# =============================
with col2:
    st.subheader("🔬 Auto-Tune RAG")
    run_button = st.button("Run Auto-Tune (parallel)", disabled=not bool(files))

    results: List[Tuple[RagConfig, float, Dict]] = []

    if run_button and files:
        with st.spinner("Running experiments (threads)…"):
            # Build grid (keep small for demo speed)
            grid: List[RagConfig] = []
            for cs in st.session_state["chunk_sizes"]:
                for ov in st.session_state["overlaps"]:
                    for tk in st.session_state["topk_values"]:
                        for rr in st.session_state["use_rerank"]:
                            for ps in ["tight", "verbose"]:
                                grid.append(RagConfig(cs, ov, tk, rr == "on", ps))
            # Trim to 4-8 combos to keep it snappy
            grid = grid[:6] if len(grid) > 6 else grid

            # Capture credentials once
            openai_key = st.session_state.get("openai_key", "") or os.getenv("OPENAI_API_KEY", "")
            groq_key = st.session_state.get("groq_key", "") or os.getenv("GROQ_API_KEY", "")

            # Build a single LLM instance per thread task (we'll create inside worker lambda)
            nq = st.session_state.get("n_questions", 5)  # safely fetch once at call site

            def run_eval(cfg: RagConfig):
                try:
                    llm_local = get_llm(provider, openai_key, groq_key, temperature=0.0)
                    avg, detail = evaluate_config(llm_local, cfg, all_docs, embed_model_name, nq)
                    return cfg, avg, detail
                except Exception as e:
                    return cfg, None, f"ERROR: {e}"

            results = []
            best_so_far = -1.0
            EARLY_STOP_THRESHOLD = 0.88
            total = len(grid)
            done = 0
            progress = st.progress(0.0, text="Evaluating configs…")
            print("darbar")
            with ThreadPoolExecutor(max_workers=min(4, len(grid))) as ex:
                futures = {ex.submit(run_eval, cfg): cfg for cfg in grid}
                for fut in as_completed(futures):
                    cfg = futures[fut]
                    try:
                        cfg, avg, detail = fut.result()
                    except Exception as e:
                        done += 1
                        progress.progress(done/total, text=f"Evaluated {done}/{total} (error)")
                        continue
                    print(f)
                    results.append((cfg, avg, detail))
                    if avg is not None and avg > best_so_far:

                        best_so_far = avg
                    done += 1
                    progress.progress(done/total, text=f"Evaluated {done}/{total}")
                    if best_so_far is not None and best_so_far >= EARLY_STOP_THRESHOLD:
 
                        break
        print(f"vali{results}")
        if results:
            results = [r for r in results if r[1] is not None]

            if not results:
                st.error("❌ All Auto-Tune evaluations failed. Check API keys / model / network.")
                st.stop()

            results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
            st.success("Auto-tune completed.")
            st.write("### 🏁 Leaderboard (avg faithfulness)")
            for i, (cfg, score, _) in enumerate(results_sorted, start=1):
                st.write(
                    f"**{i}.** score: **{score:.3f}** — chunks={cfg.chunk_size}/{cfg.overlap}, top_k={cfg.top_k}, rerank={'on' if cfg.rerank else 'off'}, prompt={cfg.prompt_style}"
                )

            best_cfg, best_score, best_detail = results_sorted[0]
            st.info(f"Best config selected → score {best_score:.3f}")

            # Persist to session
            print("hello")
            st.session_state.best_cfg = best_cfg
            st.session_state.embed_model_name = embed_model_name
            st.session_state.raw_docs = all_docs
            st.session_state.provider = provider
            # Keep whatever keys the user entered earlier
            st.session_state.openai_key = st.session_state.get("openai_key", openai_key)
            st.session_state.groq_key = st.session_state.get("groq_key", groq_key)

            with st.expander("See evaluation samples"):
                for r in best_detail["runs"][:5]:
                    st.markdown(f"**Q:** {r['q']} **Score:** {r['score']:.2f} **Ans (truncated):** {r['answer'][:400]}…")

# =============================
# Chat with the Best Config + Memory + Hybrid Retrieval
# =============================
st.subheader("💬 Chat with the Best Config + Memory")
if hasattr(st.session_state, "best_cfg"):
    best_cfg: RagConfig = st.session_state.best_cfg

    splitter = build_splitter(best_cfg)
    chunks = splitter.split_documents(st.session_state.get("raw_docs", []))

    if not chunks:
        st.warning("No chunks found. Upload PDFs and run Auto-Tune first.")
    else:
        embeddings = embedder(st.session_state["embed_model_name"]) \
            if st.session_state.get("embed_model_name") else embedder(embed_model_name)
        vectordb = FAISS.from_documents(chunks, embeddings)

        chat_llm = get_llm(
            st.session_state.get("provider", provider),
            st.session_state.get("openai_key", os.getenv("OPENAI_API_KEY", "")),
            st.session_state.get("groq_key", os.getenv("GROQ_API_KEY", "")),
            temperature=0.1,
        )

        # Minimal chat memory (own list)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history: List[Tuple[str, str]] = []  # (role, text)

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        user_q = st.chat_input("Ask a question about your documents… (memory on)")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            with st.chat_message("user"):
                st.markdown(user_q)

            # Vector search
            docs = vectordb.similarity_search(user_q, k=best_cfg.top_k)
            docs = rerank_if_enabled(user_q, docs, best_cfg.rerank)

            # Optional graph retrieval
            graph_context = ""
            if st.session_state.get("use_graph"):
                try:
                    from langchain_community.graphs import Neo4jGraph
                    graph = Neo4jGraph(url=st.session_state["neo4j_uri"], username=st.session_state["neo4j_user"], password=st.session_state["neo4j_pwd"])
                    import re
                    terms = re.findall(r"[A-Za-z][A-Za-z0-9_]+", user_q)
                    terms = [t for t in terms if len(t) > 2][:5]
                    cy = " OR ".join([f"n.name CONTAINS '{t}'" for t in terms]) or "TRUE"
                    cypher = f"MATCH (n)-[r]->(m) WHERE {cy} RETURN n.name AS s, type(r) AS r, m.name AS t LIMIT 10"
                    rows = graph.query(cypher)
                    if rows:
                        graph_context = " ".join([f"Graph: {r['s']} -[{r['r']}]-> {r['t']}" for r in rows])
                except Exception:
                    graph_context = ""

            vec_ctx = " ".join(f"[C{i+1}] " + d.page_content for i, d in enumerate(docs))
            full_ctx = (vec_ctx + "\n\n" + graph_context).strip()

            # Build messages with lightweight memory
            history_snippets = []
            for role, msg in st.session_state.chat_history[-6:]:  # last 6 turns
                prefix = "User:" if role == "user" else "Assistant:"
                history_snippets.append(f"{prefix} {msg}")
            history_text = "\n".join(history_snippets)

            sys = SystemMessage(content=(
                "You are a helpful RAG assistant. Prefer the provided context. "
                "If the answer is not contained in context, say 'I don't know'. "
                "Cite using [C#] markers that refer to provided chunks."
            ))
            human = HumanMessage(content=(
                f"Conversation so far:\n{history_text}\n\nContext:\n{full_ctx}\n\nQuestion: {user_q}\nAnswer with citations where relevant:"
            ))

            raw = chat_llm.invoke([sys, human])
            answer = to_text(raw)

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, d in enumerate(docs, start=1):
                        meta = d.metadata if isinstance(d.metadata, dict) else {}
                        loc = f"p.{meta.get('page', '?')}"
                        st.markdown(f"**[C{i}]** {loc} — {d.page_content[:400]}…")
                    if graph_context:
                        st.markdown("**Graph Facts:** " + graph_context)

            st.session_state.chat_history.append(("assistant", answer))
else:
    st.info("Run Auto-Tune to enable chat.")
