# app/main.py  (replace your existing module)
import os
import json
import gc
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

# LangChain / vectorstore imports (kept as in your code)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# ---------------- App ----------------
app = FastAPI(title="ABAP Code Explanation API")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- In-memory job tracker ---
# JOBS[job_id] = {"status": "pending"|"running"|"done"|"failed", "result": [...], "error": "..."}
JOBS: Dict[str, Dict[str, Any]] = {}

# ---- Strict input models ----
class ABAPSnippet(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    code: str

    @field_validator("code", mode="before")
    @classmethod
    def clean_code(cls, v):
        return v.strip() if v else v


# ---- Summarizer ----
def summarize_snippet(snippet: ABAPSnippet) -> dict:
    return {
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "unit_type": snippet.type,
        "unit_name": snippet.name,
        "code": snippet.code,
    }


# --- Load RAG knowledge base (constructed once at import) ---
rag_file_path = os.path.join(os.path.dirname(__file__), "rag_knowledge_base.txt")
if not os.path.exists(rag_file_path):
    logger.warning("RAG knowledge base file not found at %s", rag_file_path)

loader = TextLoader(file_path=rag_file_path, encoding="utf-8")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Vectorstore
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()


def cleanup_memory(*args):
    for var in args:
        try:
            del var
        except Exception:
            pass
    gc.collect()


def build_chain(snippet: ABAPSnippet):
    """
    Build the LangChain prompt/llm/parser pipeline and return (chain, retrieved_context).
    This function is synchronous and returns LangChain chain — *invoking* the chain can be blocking.
    """
    # NOTE: use the retriever API to fetch docs for the snippet.code
    # depending on the vectorstore wrapper, the method name might differ; using get_relevant_documents if available
    # fallback to .get_documents or .similarity_search depending on implementation.
    retrieved_docs = []
    try:
        # Preferred LangChain retriever API
        if hasattr(retriever, "get_relevant_documents"):
            retrieved_docs = retriever.get_relevant_documents(snippet.code)
        elif hasattr(retriever, "get_relevant_documents_async"):
            # synchronous path: call sync variant if present
            retrieved_docs = retriever.get_relevant_documents_async(snippet.code)
        elif hasattr(retriever, "get_relevant_documents_by_query"):
            retrieved_docs = retriever.get_relevant_documents_by_query(snippet.code)
        elif hasattr(retriever, "similarity_search"):
            retrieved_docs = retriever.similarity_search(snippet.code)
        else:
            # last-resort: try `invoke` like in your earlier attempts, but this may not exist
            if hasattr(retriever, "invoke"):
                retrieved_docs = retriever.invoke(snippet.code)
            else:
                logger.warning("Retriever does not expose a known retrieval method; proceeding without context")
                retrieved_docs = []
    except Exception as exc:
        logger.exception("Error while retrieving RAG docs: %s", exc)
        retrieved_docs = []

    retrieved_context = "\n\n".join([getattr(d, "page_content", str(d)) for d in retrieved_docs])

    SYSTEM_MSG = (
        "You are a precise ABAP reviewer and explainer. "
        "For every select query you should first list down all fields, "
        "tables and where condition. Respond in strict JSON only."
    )

    USER_TEMPLATE = """
You are an SAP ABAP Developer with 20 years of experience.
Make sure to **Have all fields of select query with all its conditions. And also all the different conditions in Code**
Based on the RAG context and ABAP code, generate a complete and professionally 
formatted explanation.

Return ONLY strict JSON:
{{
  "explanation": "<concise explanation of ABAP code>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context (from knowledge base):
{retrieved_context}

Snippet JSON:
{context_json}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()
    # Compose chain-like object (your environment used operator | previously)
    chain = prompt | llm | parser
    return chain, retrieved_context


def llm_explain_sync(snippet: ABAPSnippet):
    """
    Synchronous wrapper that builds chain and invokes it.
    This is potentially blocking and should be run in a thread (asyncio.to_thread) when used in async code.
    """
    chain, retrieved_context = build_chain(snippet)
    ctx_json = json.dumps(summarize_snippet(snippet), ensure_ascii=False, indent=2)
    invoke_payload = {
        "context_json": ctx_json,
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "type": snippet.type,
        "name": snippet.name,
        "retrieved_context": retrieved_context,
    }
    # chain.invoke is synchronous/blocking in many LangChain wrappers
    return chain.invoke(invoke_payload)


# 🧠 Background Task
async def process_explanation_job(job_id: str, snippets: List[ABAPSnippet]):
    """Runs the ABAP explanation job asynchronously without blocking the event loop."""
    JOBS[job_id]["status"] = "running"
    try:
        results = []

        # Process snippets sequentially but offload heavy LLM calls to a thread pool
        for snippet in snippets:
            try:
                # Run the blocking chain invoke in a thread to keep event loop responsive
                llm_result = await asyncio.to_thread(llm_explain_sync, snippet)
                explanation = llm_result.get("explanation", "") if isinstance(llm_result, dict) else str(llm_result)
            except Exception as e_snip:
                logger.exception("Error explaining snippet: %s", e_snip)
                explanation = f"[Error during LLM explain: {e_snip}]"

            results.append({
                "pgm_name": snippet.pgm_name,
                "inc_name": snippet.inc_name,
                "type": snippet.type,
                "name": snippet.name,
                "code": "",
                "purpose": explanation,
            })

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = results

    except Exception as e:
        logger.exception("Job failed: %s", e)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.post("/explain-abap")
async def explain_abap(snippets: List[ABAPSnippet]):
    """
    Starts a background ABAP explanation job.
    Returns a job_id that can be used to poll status or results.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "result": None, "error": None}

    # Start the job asynchronously and return immediately
    # Use create_task so the job runs independently of current request/connection
    asyncio.create_task(process_explanation_job(job_id, snippets))

    return {"job_id": job_id, "status": "started"}


@app.get("/explain-abap/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status or get final results."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Invalid job_id")

    if job["status"] == "pending" or job["status"] == "running":
        return {"status": job["status"]}
    elif job["status"] == "failed":
        return {"status": "failed", "error": job.get("error")}
    elif job["status"] == "done":
        return job.get("result", [])
    else:
        return {"status": "unknown"}


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
