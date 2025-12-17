# src/pipeline.py
import os
from src.retriever import Retriever

# Optional: Load env vars if you have a .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# System message instructing the LLM on its role and rules
SYSTEM_PROMPT = (
    "You are an expert Q&A assistant. Your goal is to answer the user's question "
    "FACTUALLY and CONCISELY based ONLY on the context provided. "
    "For each paragraph, end with: (Source: filename)"
    "If the context does not contain the answer, you MUST respond with: "
    "'The provided context does not contain the answer to this question.'"
)

# LLM Setup
try:
    from openai import OpenAI, RateLimitError
    
    # GROQ SETUP
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY") 
    )
    
    def llm_answer(prompt, model="llama-3.1-8b-instant"):
        if not client.api_key:
            raise ValueError("No API Key found")
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            resp = client.chat.completions.create(
                model=model, 
                messages=messages 
            )
            return resp.choices[0].message.content
        except RateLimitError:
            return "⚠️ GROQ ERROR: Rate limit reached. Try again in a minute."
        except Exception as e:
            return f"LLM Error: {e}"

except Exception as e:
    # Fallback if OpenAI is missing or fails
    client = None
    def llm_answer(prompt, model=None):
        return f"[LLM Mock Output] API Key missing or Error: {e}.\n\nRetrieval Context Preview:\n" + prompt[-500:]

def make_prompt(question, passages):
    # Combine sources into a context block
    context = "\n\n---\n\n".join([f"Source: {p.get('source', 'Unknown')}\nText: {p['text']}" for p in passages])
    
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    return prompt

class RAG:
    def __init__(self):
        self.ret = Retriever(
            remove_stopwords=True,
            use_stemming=True
        )

    def answer(self, question, top_k=5):
        passages = passages = self.ret.hybrid(
        question, top_k=top_k, bm25_k=40, dense_k=40
    )
        prompt = make_prompt(question, passages)
        answer = llm_answer(prompt)
        return {"answer": answer, "passages": passages}

if __name__ == "__main__":
    rag = RAG()
    
    query = "Differnce in BM25 and Dense retrieval?"   #Default query for testing
    print(f"Query: {query}")
    print("-" * 30)
    
    out = rag.answer(query)
    
    print("Generated Answer:")
    print(out["answer"])
    print("-" * 30)
    print("Sources Used:")
    for p in out["passages"][:3]:
        print(f"-> [{p.get('source', 'doc')}] {p['id']} (Score: {p.get('score', 0):.4f})")