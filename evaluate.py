import os
import json
import time
import re
import pandas as pd
from datetime import datetime

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator
from app_config import get_dual_models
from research_engine import get_research_engine
from agent_logic import get_agentic_engine

def extract_json_from_text(text):
    """Senior Engineer Fix: Uses Regex to find and extract ONLY the JSON object."""
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None

def generate_test_data(num_q):
    print(f"\n🚀 GENERATING {num_q} QUESTIONS USING GROQ (70B)...")
    try:
        reasoning_llm, _, _ = get_dual_models()
        engine = get_research_engine()
        
        nodes = engine.retriever.retrieve("summary of methodology and results")
        
        test_data =[]
        print("🤖 Analyzing documents to create Q&A pairs...")
        
        for i, node in enumerate(nodes[:num_q]):
            prompt = (
                "You are an academic professor. Read the following text and write exactly ONE "
                "research question and ONE correct answer (Ground Truth) based ONLY on this text.\n"
                "OUTPUT FORMAT: You must output ONLY a valid JSON object. Do not use markdown backticks. "
                "Do not add introductory text. Use this exact structure:\n"
                '{"query": "your question", "reference": "your ground truth answer"}\n\n'
                f"TEXT:\n{node.get_content()}"
            )
            
            response = reasoning_llm.complete(prompt)
            json_data = extract_json_from_text(str(response)) # <-- Regex Sanitizer in action!
            
            if json_data and "query" in json_data and "reference" in json_data:
                test_data.append(json_data)
                print(f"  ✓ Generated Q{i+1}: {json_data['query'][:50]}...")
            else:
                print(f"  ⚠️ Skipped node {i+1} due to formatting issues.")
            
            time.sleep(5) # Throttle to protect Groq limits
        
        with open("evaluation_testset.json", "w") as f:
            json.dump(test_data, f, indent=4)
            
        print(f"\n✅ Success! Created evaluation_testset.json with {len(test_data)} pairs.")

    except Exception as e:
        print(f"❌ Error during generation: {e}")

def run_evaluation_suite():
    print("\n🚀 STARTING GROQ AGENT AUDIT...")
    if not os.path.exists("evaluation_testset.json"):
        print("❌ Error: Run --generate first.")
        return

    try:
        with open("evaluation_testset.json", "r") as f:
            test_queries = json.load(f)

        reasoning_llm, _, _ = get_dual_models()
        agent = get_agentic_engine()
        
        faith_eval = FaithfulnessEvaluator(llm=reasoning_llm)
        rel_eval = RelevancyEvaluator(llm=reasoning_llm)
        corr_eval = CorrectnessEvaluator(llm=reasoning_llm)

        eval_results =[]

        for entry in test_queries:
            query = entry["query"]
            reference = entry["reference"] 
            print(f"\nTesting Agent on: {query[:50]}...")
            
            response = agent.query(query)
            
            f_result = faith_eval.evaluate_response(response=response)
            r_result = rel_eval.evaluate_response(query=query, response=response)
            c_result = corr_eval.evaluate_response(query=query, response=response, reference=reference)
            
            eval_results.append({
                "Question": query,
                "Faithfulness": 1 if f_result.passing else 0,
                "Relevancy": 1 if r_result.passing else 0,
                "Correctness": c_result.score,
                "Agent_Answer": str(response),
                "Ground_Truth": reference
            })
            
            time.sleep(6) 

        df = pd.DataFrame(eval_results)
        print("\n" + "="*40)
        print("🏆 LLAAM 3.3 RAG AUDIT REPORT")
        print("="*40)
        print(f"Avg Faithfulness: {df['Faithfulness'].mean():.2%}")
        print(f"Avg Relevancy:     {df['Relevancy'].mean():.2%}")
        print(f"Avg Correctness:   {df['Correctness'].mean():.1f}/5.0")
        print("="*40)
        
        df.to_csv("groq_audit_report.csv", index=False)
        print("Saved to groq_audit_report.csv")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to generate")
    args = parser.parse_args()
    
    if args.generate: 
        generate_test_data(args.n)
    elif args.evaluate: 
        run_evaluation_suite()