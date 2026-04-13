import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

def test_groq():
    print("Connecting to Groq LPU...")
    try:
        # We use the instant model for testing
        llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        response = llm.complete("Hi Groq! Explain your speed in 10 words.")
        print(f"\nResponse: {response}")
        print("\n✅ Phase 1 Successful: Ready for Dual-Model RAG!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_groq()