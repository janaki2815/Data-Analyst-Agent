📊 Data Analyst (SQL + AI Agent)

A simple AI-powered data analysis tool built with Streamlit, DuckDB, and Groq/OpenAI.
Upload any CSV or Excel file → ask questions in natural language or SQL → get instant results.

🌟 Features

Upload CSV/XLSX files

Run SQL queries directly on your data (DuckDB)

Ask natural-language questions (AI converts them to SQL)

Supports Groq & OpenAI models

Robust CSV/XLSX parsing (multiple encodings & delimiters)

Automatic model fallback for safer and more accurate responses

Download results as CSV

Simple and clean interface

🚀 How It Works

Upload a dataset

The file is loaded into DuckDB as a table named:

uploaded_data


You can:

Write SQL like:

SELECT COUNT(*) FROM uploaded_data;


Or ask:

Show rows with missing emails.


The app returns the results instantly.

🛠️ Tech Stack

Python

Streamlit (UI)

DuckDB (SQL Engine)

Pandas

Groq / OpenAI LLMs

AGNO Agent Framework

🔧 Installation
git clone https://github.com/janaki2815/Data-Analyst-Agent.git
cd Data-Analyst-Agent
pip install -r requirements.txt

▶️ Run the App
streamlit run app_universal_data_agent.py


Open in browser:

http://localhost:8501

🔑 API Keys

Paste your keys in the sidebar:

Groq API Key (recommended)

OpenAI API Key (optional)

No .env required.

📌 Example SQL Queries
SELECT * FROM uploaded_data LIMIT 10;
SELECT COUNT(*) FROM uploaded_data;
SELECT DISTINCT Email FROM uploaded_data;
SELECT Name, COUNT(*) FROM uploaded_data GROUP BY Name;

🧠 Example Natural Language Queries

"Show duplicate emails"

"Get the top 10 most common names"

"Count rows where phone number is missing"

"Summarize the dataset"

📂 Project Structure
app_universal_data_agent.py   # main app
README.md
requirements.txt
