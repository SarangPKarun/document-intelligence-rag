import json
import os
import pandas as pd
import sys
import traceback
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set default env vars for local execution if not present
if "WEAVIATE_URL" not in os.environ:
    os.environ["WEAVIATE_URL"] = "http://localhost:8080"
if "OLLAMA_BASE_URL" not in os.environ:
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

from app.agent import app_agent, ingest_text

# Configuration
LLM_MODEL = "smollm2:1.7b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_judge_llm():
    print(f"Using Local Ollama ({LLM_MODEL}) for evaluation.")
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

def grade_answer(question, answer, ground_truth, context):
    llm = get_judge_llm()
    
    # --- 1. Retrieval Metrics ---
    
    # A. Retrieval Accuracy (Recall): Does the context contain the answer found in ground truth?
    retrieval_acc_prompt = PromptTemplate.from_template("""
    You are a judge evaluating retrieval quality.
    Question: {question}
    Correct Answer (Ground Truth): {ground_truth}
    Retrieved Context: {context}
    
    Does the Retrieved Context contain the information necessary to answer the question as per the Ground Truth?
    Rate from 1 to 5.
    1 = Not at all (Context is irrelevant or missing info)
    3 = Partially (Some info is there, but some missing)
    5 = Fully (All necessary info is present)
    
    Output ONLY the number (1-5).
    """)
    try:
        score = (retrieval_acc_prompt | llm | StrOutputParser()).invoke({
            "question": question, "ground_truth": ground_truth, "context": context
        }).strip()
        retrieval_accuracy = int(''.join(filter(str.isdigit, score))[0])
    except:
        retrieval_accuracy = 0

    # B. Retrieval Precision: Is the context mostly relevant or full of noise?
    retrieval_prec_prompt = PromptTemplate.from_template("""
    You are a judge evaluating retrieval precision.
    Question: {question}
    Retrieved Context: {context}
    
    How much of the Retrieved Context is actually relevant to the Question?
    Rate from 1 to 5.
    1 = Mostly Noise (Most retrieved text is irrelevant)
    3 = Mixed (Half relevant, half noise)
    5 = Highly Precise (Almost all text is relevant)
    
    Output ONLY the number (1-5).
    """)
    try:
        score = (retrieval_prec_prompt | llm | StrOutputParser()).invoke({
            "question": question, "context": context
        }).strip()
        retrieval_precision = int(''.join(filter(str.isdigit, score))[0])
    except:
        retrieval_precision = 0

    # --- 2. Contextual Metrics (Generation) ---

    # A. Contextual Accuracy (formerly Correctness): Answer vs Ground Truth
    contextual_acc_prompt = PromptTemplate.from_template("""
    You are a strict teacher grading an exam. 
    Question: {question}
    Student Answer: {answer}
    Correct Answer (Ground Truth): {ground_truth}
    
    Grade the Student Answer from 1 to 5 based on how well it matches the Correct Answer.
    1 = Completely wrong
    3 = Partially correct
    5 = Completely correct
    
    Output ONLY the number (1-5).
    """)
    try:
        score = (contextual_acc_prompt | llm | StrOutputParser()).invoke({
            "question": question, "answer": answer, "ground_truth": ground_truth
        }).strip()
        contextual_accuracy = int(''.join(filter(str.isdigit, score))[0])
    except:
        contextual_accuracy = 0

    # B. Contextual Precision (formerly Relevance): Answer vs Question
    contextual_prec_prompt = PromptTemplate.from_template("""
    You are a judge evaluating if an answer is relevant to the question.
    Question: {question}
    Answer: {answer}
    
    Rate relevance from 1 to 5.
    1 = Irrelevant / Hallucination
    3 = Somewhat relevant
    5 = Highly relevant / Direct answer
    
    Output ONLY the number (1-5).
    """)
    try:
        score = (contextual_prec_prompt | llm | StrOutputParser()).invoke({
            "question": question, "answer": answer
        }).strip()
        contextual_precision = int(''.join(filter(str.isdigit, score))[0])
    except:
        contextual_precision = 0
        
    return retrieval_accuracy, retrieval_precision, contextual_accuracy, contextual_precision

def evaluate():
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL}")

    # 1. Ingest Data
    print("Ingesting sample_rag.txt...")
    try:
        with open('sample_rag.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            # Ingest
            ingest_text(text, source="sample_rag.txt")
            print("Ingestion successful.")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        # Proceeding might be futile but let's try
    
    # 2. Load Dataset
    with open('evaluation/test_dataset.json', 'r') as f:
        data = json.load(f)
        
    results = []
    
    print("Starting evaluation loop...")
    for item in data:
        q = item['question']
        gt = item['ground_truth']
        
        print(f"Processing Q: {q}")
        
        # Get Agent Answer
        try:
            response = app_agent.invoke({"question": q})
            ans = response['answer']
            ctx = response.get('context', [])
            ctx_str = "\n".join(ctx)
        except Exception as e:
            print(f"Agent failed: {e}")
            ans = "ERROR"
            ctx_str = ""
            
        # Grade
        r_acc, r_prec, c_acc, c_prec = grade_answer(q, ans, gt, ctx_str)
        print(f"  -> Scores: Ret_Acc={r_acc}, Ret_Prec={r_prec}, Ctx_Acc={c_acc}, Ctx_Prec={c_prec}")
        
        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "context_retrieved": ctx_str,
            "retrieval_accuracy": r_acc,
            "retrieval_precision": r_prec,
            "contextual_accuracy": c_acc,
            "contextual_precision": c_prec
        })

    # 3. Save Results
    df = pd.DataFrame(results)
    print("\nAverage Scores:")
    print(df[['retrieval_accuracy', 'retrieval_precision', 'contextual_accuracy', 'contextual_precision']].mean())
    
    df.to_csv('evaluation/results_custom.csv', index=False)
    print("Saved to evaluation/results_custom.csv")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception:
        traceback.print_exc()
