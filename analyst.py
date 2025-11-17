# app_universal_data_agent.py
import os
import io
import tempfile
import csv
import streamlit as st
import pandas as pd
from time import sleep
import requests
import duckdb

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# -----------------------
# Config: safe models & blacklist
# -----------------------
SAFE_MODELS = [
    "llama-3.3-70b",
    "llama-4-scout",
    "kimi-k2",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-3.5-turbo",
]

BLOCKED_MODEL_KEYWORDS = [
    "safeguard", "prompt-guard", "guard", "protector",
    "tts", "whisper", "maverick", "vision", "audio",
    "embed", "playai", "text-to-speech", "speech", "translate"
]

# -----------------------
# Utilities: model & agent helpers
# -----------------------
def is_blocked_model(model_id: str) -> bool:
    if not model_id:
        return False
    mid = model_id.lower()
    return any(k in mid for k in BLOCKED_MODEL_KEYWORDS)

def create_openaichat_wrapper(model_id, api_key, base_url=None):
    model_kwargs = {"id": model_id, "api_key": api_key}
    if base_url:
        model_kwargs["base_url"] = base_url
    try:
        return OpenAIChat(**model_kwargs)
    except TypeError:
        return OpenAIChat(id=model_id, api_key=api_key)

def build_agent_for_model(model_id, api_key, base_url=None):
    mw = create_openaichat_wrapper(model_id, api_key, base_url)
    return Agent(
        model=mw,
        tools=[DuckDbTools(), PandasTools()],
        system_message=(
            "You are an expert data analyst. Use the DuckDB table 'uploaded_data' to answer queries. "
            "When appropriate, produce SQL statements or tabular outputs in a parseable format (markdown tables or CSV)."
        ),
        markdown=True,
    )

def list_groq_models(api_key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [m.get("id") for m in data.get("data", []) if m.get("id")]

# -----------------------
# Robust CSV/XLSX reading
# -----------------------
def robust_read_and_save(uploaded_file):
    """Try multiple encodings/delimiters and return (tmp_csv_path, columns, df) or (None,None,None)."""
    content = uploaded_file.getvalue()
    encodings = ["utf-8", "latin1", "cp1252"]
    delimiters = [",", ";", "\t", "|"]
    for enc in encodings:
        for delim in delimiters:
            try:
                buf = io.BytesIO(content)
                df = pd.read_csv(buf, encoding=enc, sep=delim, engine="python", na_values=['NA','N/A','missing'])
                if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                    # sanitize string columns
                    for col in df.select_dtypes(include=['object']):
                        df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
                        return tmp.name, df.columns.tolist(), df
            except Exception:
                continue
    # fallback: try pandas read_table
    try:
        buf = io.BytesIO(content)
        df = pd.read_table(buf, engine="python", na_values=['NA','N/A','missing'])
        if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
                return tmp.name, df.columns.tolist(), df
    except Exception:
        pass
    return None, None, None

def preprocess_and_save_default(file):
    """Simple reader (used as fallback)."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            return None, None, None
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        return temp_path, df.columns.tolist(), df
    except Exception:
        return None, None, None

# -----------------------
# Agent run retries
# -----------------------
def run_agent_with_retries(agent, prompt, attempts=3, initial_delay=2):
    delay = initial_delay
    last_exc = None
    for i in range(attempts):
        try:
            return agent.run(prompt)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # retry for rate/quota issues
            if "quota" in msg or "rate limit" in msg or "429" in msg or "insufficient_quota" in msg:
                if i < attempts - 1:
                    st.warning("Rate/Quota error from provider — retrying...")
                    sleep(delay)
                    delay *= 2
                    continue
                else:
                    raise
            else:
                raise
    raise last_exc

# -----------------------
# DuckDB helpers (file-backed DB)
# -----------------------
def ensure_duckdb_table_from_csv(csv_path, db_file):
    """Create (or replace) uploaded_data table in db_file using read_csv_auto."""
    # normalize path for SQL (POSIX style)
    csv_posix = csv_path.replace("\\", "/")
    conn = duckdb.connect(database=db_file)
    # Use read_csv_auto to let DuckDB detect delimiter/encoding
    create_sql = f"""
    CREATE OR REPLACE TABLE uploaded_data AS
    SELECT * FROM read_csv_auto('{csv_posix}');
    """
    conn.execute(create_sql)
    # verify
    row_count = conn.execute("SELECT COUNT(*) FROM uploaded_data").fetchone()[0]
    sample_df = conn.execute("SELECT * FROM uploaded_data LIMIT 5").fetchdf()
    conn.close()
    return row_count, sample_df

def run_sql_and_show(db_file, sql):
    conn = duckdb.connect(database=db_file)
    try:
        df = conn.execute(sql).fetchdf()
    finally:
        conn.close()
    return df

# -----------------------
# Streamlit UI: minimal & clear
# -----------------------
st.set_page_config(page_title="Universal Data Agent", layout="wide")
st.title("📊 Universal Data Analyst (SQL + NL)")

with st.sidebar:
    st.header("API Keys")
    groq_key = st.text_input("GROQ API Key (recommended)", type="password")
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")

    st.markdown("---")
    st.markdown("Model selection (override auto-pick):")
    model_id = st.selectbox("Model", options=SAFE_MODELS, index=0)

    st.markdown("---")
    st.write("Notes:")
    st.write("- Paste keys only locally. Keys are stored in the session only.")
    st.write("- For raw SQL queries start with SELECT/PRAGMA/WITH etc.; these run directly via DuckDB.")

# Upload widget
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

if not (groq_key or openai_key):
    st.warning("Please paste a GROQ or OpenAI key in the sidebar to use the agent (SQL alone still works).")

# Read file robustly
temp_path, columns, df = robust_read_and_save(uploaded_file)
if temp_path is None:
    temp_path, columns, df = preprocess_and_save_default(uploaded_file)

# Debug info
st.markdown("**File debug info**")
st.write("Uploaded filename:", uploaded_file.name)
st.write("Temp CSV path:", temp_path)
try:
    st.write("File exists on disk:", os.path.exists(temp_path))
except Exception as ex:
    st.write("os.path.exists error:", ex)
if df is None:
    st.error("Failed to parse file into a DataFrame. Try re-saving CSV or upload a small sample.")
    st.stop()
st.write("DataFrame shape:", df.shape)
st.write("Columns:", columns)
st.dataframe(df.head(8))

# Ensure table exists in a file-backed DuckDB DB that both agent and direct SQL can use
db_file = os.path.join(tempfile.gettempdir(), "universal_uploaded_data.duckdb")

try:
    st.info("Creating 'uploaded_data' table inside a persistent DuckDB file...")
    row_count, sample_df = ensure_duckdb_table_from_csv(temp_path, db_file)
    st.success(f"Table 'uploaded_data' ready in DuckDB ({db_file}); rows: {row_count}")
    st.write("Sample (first 5 rows):")
    st.dataframe(sample_df)
except Exception as e:
    st.error(f"Failed to create DuckDB table from CSV: {e}")
    st.stop()

# Build Agent if user provided a key (otherwise agent operations disabled)
using_groq = bool(groq_key)
chosen_key = groq_key if using_groq else openai_key
chosen_base_url = "https://api.groq.com/openai/v1" if using_groq else None

if chosen_key:
    # sanitize user-chosen model
    if is_blocked_model(model_id):
        st.warning(f"Selected model '{model_id}' is blocked. Switching to a safe default.")
        model_id = SAFE_MODELS[0]

    # if groq, attempt auto-pick based on visible models
    if using_groq:
        try:
            visible = list_groq_models(groq_key)
            visible_good = [m for m in visible if not is_blocked_model(m)]
            pick = None
            for m in SAFE_MODELS:
                if m in visible_good:
                    pick = m
                    break
            if pick:
                if pick != model_id:
                    st.info(f"Auto-selected GROQ model '{pick}' based on your key.")
                    model_id = pick
            else:
                if visible_good:
                    model_id = visible_good[0]
                    st.info(f"Using first available non-blocked model: {model_id}")
                else:
                    st.warning("No non-blocked models visible to Groq key; proceeding with chosen model.")
        except Exception as e:
            st.warning(f"Could not list Groq models: {e} (proceeding)")

    try:
        agent = build_agent_for_model(model_id, chosen_key, chosen_base_url)
        st.success(f"Agent initialized with model: {model_id}")
    except Exception as e:
        st.error(f"Failed to initialize agent with model '{model_id}': {e}")
        agent = None
else:
    agent = None
    st.info("No API key provided — agent (natural-language) is disabled. You can still run raw SQL queries directly.")

# Query area
st.markdown("---")
st.subheader("Run a query")
user_query = st.text_area("Enter SQL (starting with SELECT) or natural language question for the agent:", value="SELECT * FROM uploaded_data LIMIT 5;", height=160)

def looks_like_sql(text: str) -> bool:
    if not text:
        return False
    k = text.strip().split()[0].upper()
    return k in ("SELECT", "WITH", "PRAGMA", "SHOW", "DESCRIBE", "EXPLAIN", "CREATE", "DROP", "ALTER")

if st.button("Run"):
    if not user_query or user_query.strip() == "":
        st.warning("Enter a SQL query or question.")
    else:
        # If the user input looks like SQL, run directly in DuckDB (guaranteed to work if table exists)
        if looks_like_sql(user_query):
            try:
                df_out = run_sql_and_show(db_file, user_query)
                st.write("SQL result:")
                st.dataframe(df_out)
                # allow download
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download result CSV", data=csv_bytes, file_name="query_result.csv")
            except Exception as e:
                st.error(f"Error running SQL: {e}")
        else:
            # Use agent to interpret NL -> SQL and run. Require agent available.
            if agent is None:
                st.error("No agent available: paste a GROQ or OpenAI key in the sidebar to enable natural-language queries.")
            else:
                try:
                    with st.spinner("Agent is generating and running SQL..."):
                        # run via the AGNO agent; it should use DuckDB tools available to it
                        response = run_agent_with_retries(agent, user_query, attempts=3)
                        content = getattr(response, "content", str(response))
                    st.markdown("**Agent response:**")
                    st.markdown(content)
                    # Try to extract SQL from response (simple heuristic) and run it — optional
                    # If agent included an explicit SQL code block, run it automatically
                    import re
                    sql_blocks = re.findall(r"```sql\n(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
                    if not sql_blocks:
                        # try fenced code block without language
                        sql_blocks = re.findall(r"```\n(.*?)```", content, flags=re.DOTALL)
                    if sql_blocks:
                        # run the first SQL block
                        try:
                            sql_to_run = sql_blocks[0].strip()
                            st.info("Detected SQL block from agent — running it and showing results.")
                            df_res = run_sql_and_show(db_file, sql_to_run)
                            st.dataframe(df_res)
                            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                            st.download_button("Download agent-query result CSV", data=csv_bytes, file_name="agent_query_result.csv")
                        except Exception as ex_sql:
                            st.warning(f"Agent produced SQL but running it failed: {ex_sql}")
                except Exception as e:
                    msg = str(e).lower()
                    # detect parse/tool calling errors and suggest safe fallback
                    if "parsing failed" in msg or "output_parse_failed" in msg or "tool calling" in msg or "not supported with this model" in msg:
                        st.error("Agent model returned unparseable output or can't call tools. Try selecting a safer model in the sidebar (e.g., gpt-3.5-turbo or llama-3.3-70b) or provide a GROQ key.")
                    elif "quota" in msg or "rate limit" in msg or "429" in msg or "insufficient_quota" in msg:
                        st.error("Quota or rate-limit error from provider.")
                    else:
                        st.error(f"Agent error: {e}")

# Helpful quick SQL snippets
st.markdown("---")
st.markdown("### Quick SQL snippets (click to copy)")
st.code("SELECT * FROM uploaded_data LIMIT 10;")
st.code("SELECT COUNT(*) AS total_rows FROM uploaded_data;")
st.code("SELECT DISTINCT Email FROM uploaded_data WHERE Email IS NOT NULL AND Email <> '' ORDER BY Email ASC;")
st.code("SELECT Name, COUNT(*) AS cnt FROM uploaded_data GROUP BY Name ORDER BY cnt DESC LIMIT 10;")
st.code("SELECT * FROM uploaded_data WHERE Email IS NULL OR Email = '';")
