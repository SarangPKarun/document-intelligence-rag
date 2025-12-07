import os
from typing import Annotated, List, TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

import weaviate

# --- Configuration ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = "smollm2:1.7b"

# --- 1. Vector Store Setup ---
embeddings = OllamaEmbeddings(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL
)

def get_vectorstore():
    return Weaviate(
        client=weaviate.Client(url=WEAVIATE_URL),
        index_name="Document",
        text_key="text",
        embedding=embeddings,
        by_text=False # relying on vector search
    )

def ingest_text(text: str, source: str):
    """Ingests raw text into Weaviate."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text], metadatas=[{"source": source}])
    
    # Initialize Weaviate and add documents
    vectorstore = Weaviate.from_documents(
        docs, 
        embeddings, 
        weaviate_url=WEAVIATE_URL, 
        index_name="Document"
    )
    return f"Ingested {len(docs)} chunks from {source}."

# --- 2. Agent State & Nodes ---

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str

# Node: Retrieve
def retrieve(state: AgentState):
    print("---RETRIEVING---")
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(state["question"])
    return {"context": [d.page_content for d in docs]}

# Node: Generate Answer
def generate(state: AgentState):
    print("---GENERATING---")
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    
    template = """
    You are an AI assistant designed to answer based on the provided context.

    Follow these rules:

    Rule 1:
    If the user greeting is like: hi, hello, hey, good morning, good evening, how are you,
    reply with a friendly greeting. Example greeting response (do NOT repeat this text, use your own):
    "Hello! I'm doing great, how can I assist you today?"

    Rule 2:
    If the answer exists in the context, respond using ONLY the context.

    Rule 3:
    If the context does not contain the answer, respond with:
    "I don't know the answer to that based on the uploaded documents."
    You may optionally add a short general polite sentence, without adding facts.

    Rule 4:
    Keep answers short, natural, and conversational. No robotic tone.

    Context:
    {context}

    User:
    {question}

    Assistant Response:
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": answer}

# --- 3. Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_agent = workflow.compile()