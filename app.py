import os
import re

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Acrux Dynamics HR Assistant", page_icon=":compass:", layout="centered")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(APP_DIR, "data")

NAME_REPLACEMENTS = [
    (r"Zyro Dynamics", "Acrux Dynamics"),
    (r"ZyroHR", "AcruxHR"),
    (r"ZyroCRM", "AcruxCRM"),
    (r"ZyroInsight", "AcruxInsight"),
    (r"ZyroDesk", "AcruxDesk"),
    (r"zyrodyn\s*amics\.com", "acruxdynamics.com"),
    (r"\bZyro\b", "Acrux"),
]


def normalize_company_name(text):
    for pattern, repl in NAME_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


RAG_PROMPT = ChatPromptTemplate.from_template('''
You are the Acrux Dynamics HR Help Desk Assistant.

Answer strictly and only using the provided HR policy context below. Do not
use outside knowledge and do not make assumptions about anything that is
not explicitly stated in the context.

Guidelines:
- Extract and include every relevant detail present in the context: numbers,
  dates, percentages, monetary amounts, durations, eligibility criteria,
  exceptions, and approval steps.
- If the question has multiple parts, answer each part explicitly and
  completely.
- Present multi-fact answers as clear, well-organised bullet points.
- Write in a natural, professional, helpful tone.
- If the context does not contain the answer, respond with exactly this
  sentence and nothing else:
  "I could not find this information in the provided HR policy documents."

Context:
{context}

Question:
{question}

Answer:
''')

OOS_PROMPT = ChatPromptTemplate.from_template('''
You are a strict scope-classifier for the Acrux Dynamics HR Help Desk
Assistant.

The assistant may ONLY answer questions that can be fully and specifically
answered using the company's internal HR policy documents (Leave Policy,
WFH Policy, Code of Conduct, Performance Review Policy, Compensation &
Benefits Policy, IT & Data Security Policy, POSH Policy, Onboarding &
Separation Policy, Travel & Expense Policy, Employee Handbook, Company
Profile).

Answer NO (out of scope) if the question asks about any of the following,
even if it loosely touches an HR topic:
- External recruitment, job application, or hiring/interview process for
  new candidates.
- A specific personal/individual number or amount that depends on an
  individual offer or manager decision rather than a documented
  company-wide policy rule.
- The company's financial performance, revenue, profit, funding, or
  valuation.
- Detailed product features of the company's software products, or any
  comparison of those products to competitor products.
- Any other company's policies or benefits, or a comparison with them.
- Anything unrelated to company HR/IT/workplace policy.

Answer YES (in scope) only if the question can be answered using rules,
numbers, eligibility criteria, or processes documented as company-wide
policy.

Question:
{question}

Respond with exactly one word: YES or NO.
''')

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."
NOT_FOUND_MARKER = "i could not find this information"


@st.cache_resource(show_spinner="Loading HR policy knowledge base...")
def load_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    for doc in documents:
        doc.page_content = normalize_company_name(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5},
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1536, groq_api_key=api_key)

    return retriever, reranker, llm


def retrieve_context(retriever, reranker, question, top_n=6):
    candidates = retriever.invoke(question)
    if not candidates:
        return []
    pairs = [[question, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def ask_bot(retriever, reranker, llm, question):
    guard_chain = OOS_PROMPT | llm | StrOutputParser()
    decision = guard_chain.invoke({"question": question}).strip().upper()

    if decision.startswith("NO") or ("NO" in decision and "YES" not in decision):
        return REFUSAL_MESSAGE, []

    docs = retrieve_context(retriever, reranker, question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    if NOT_FOUND_MARKER in answer.lower():
        return REFUSAL_MESSAGE, []

    return answer, docs


# ---------------------------------------------------------------- UI -----
st.title(":compass: Acrux Dynamics HR Assistant")
st.caption("Ask me anything about leave, WFH, compensation, performance, travel, POSH, and more.")

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers employee HR questions using Acrux Dynamics' "
        "internal policy documents. It will politely decline questions that "
        "fall outside HR policy."
    )
    st.subheader("Topics covered")
    st.markdown(
        "- Leave (EL, CL, SL, Maternity, Paternity)\n"
        "- Work From Home\n"
        "- Compensation & Benefits\n"
        "- Performance Reviews\n"
        "- IT & Data Security\n"
        "- POSH Policy\n"
        "- Onboarding & Separation\n"
        "- Travel & Expense"
    )
    st.subheader("Try asking")
    st.markdown(
        "- *How many casual leaves do I get?*\n"
        "- *What is the WFH eligibility?*\n"
        "- *What is the travel reimbursement process?*"
    )

if not os.path.isdir(CORPUS_PATH):
    st.error(
        "No 'data' folder found next to app.py. Add the 11 HR policy PDFs "
        "to a 'data/' folder in your repository before deploying."
    )
    st.stop()

retriever, reranker, llm = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(s)

question = st.chat_input("Ask an HR question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking policy documents..."):
            answer, docs = ask_bot(retriever, reranker, llm, question)
        st.markdown(answer)

        sources = []
        if docs:
            for d in docs[:4]:
                fname = os.path.basename(d.metadata.get("source", "unknown"))
                page = d.metadata.get("page", "?")
                sources.append(f"{fname} (page {page})")
            with st.expander("Sources"):
                for s in sources:
                    st.caption(s)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})