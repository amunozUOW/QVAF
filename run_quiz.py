import chromadb
import ollama
import json
import sys
from datetime import datetime

# Load the database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("unit_materials")

def search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

def extract_search_terms(question):
    response = ollama.chat(model='mistral', messages=[
        {'role': 'user', 'content': f'What are the key operations management concepts in this question? List only the technical terms, separated by commas. No explanation.\n\nQuestion: {question}'}
    ])
    return response['message']['content']

def answer_question_without_rag(question):
    """Answer using only the LLM's general knowledge"""
    response = ollama.chat(model='mistral', messages=[
        {'role': 'system', 'content': 'You are a student taking a quiz. Answer based on your general knowledge.'},
        {'role': 'user', 'content': f'''QUESTION:
{question}

Respond in this exact format:
ANSWER: [letter]
CONFIDENCE: [0-100]%
REASONING: [your step-by-step explanation]
CONFIDENCE JUSTIFICATION: [why you are confident or uncertain]'''}
    ])
    
    return {
        'answer': response['message']['content']
    }

def answer_question_with_rag(question):
    """Answer using RAG-retrieved unit materials"""
    search_terms = extract_search_terms(question)
    chunks = search(search_terms, n_results=3)
    context = "\n\n---\n\n".join(chunks)
    
    response = ollama.chat(model='mistral', messages=[
        {'role': 'system', 'content': 'You are a student taking a quiz. Use the provided unit materials to answer questions.'},
        {'role': 'user', 'content': f'''UNIT MATERIALS:
{context}

QUESTION:
{question}

Respond in this exact format:
ANSWER: [letter]
CONFIDENCE: [0-100]%
REASONING: [your step-by-step explanation]
CONFIDENCE JUSTIFICATION: [why you are confident or uncertain]'''}
    ])
    
    return {
        'answer': response['message']['content'],
        'retrieved_context': chunks,
        'search_terms': search_terms
    }

# ============================================
# MAIN
# ============================================

if len(sys.argv) < 2:
    print("Usage: python3 run_quiz.py <questions.json>")
    sys.exit(1)

questions_file = sys.argv[1]
print(f"Loading questions from {questions_file}...")

with open(questions_file, 'r') as f:
    quiz_data = json.load(f)

questions = quiz_data['questions']
quiz_name = quiz_data.get('quiz_name', 'Unnamed Quiz')

print(f"Quiz: {quiz_name}")
print(f"Running {len(questions)} questions (2 passes each: without RAG, then with RAG)...\n")

# Run the quiz
quiz_log = {
    "timestamp": datetime.now().isoformat(),
    "quiz_name": quiz_name,
    "model": "mistral",
    "questions": []
}

for i, q in enumerate(questions):
    print(f"Question {q['id']} ({i+1}/{len(questions)})...")
    
    # Pass 1: Without RAG
    print(f"  Pass 1: Without RAG...")
    result_no_rag = answer_question_without_rag(q['text'])
    
    # Pass 2: With RAG
    print(f"  Pass 2: With RAG...")
    result_with_rag = answer_question_with_rag(q['text'])
    
    quiz_log["questions"].append({
        "id": q['id'],
        "question_text": q['text'],
        "correct_answer": q['correct_answer'],
        "without_rag": {
            "agent_response": result_no_rag['answer']
        },
        "with_rag": {
            "search_terms": result_with_rag['search_terms'],
            "retrieved_context": result_with_rag['retrieved_context'],
            "agent_response": result_with_rag['answer']
        }
    })
    
    print(f"  Done.\n")

# Save to JSON
output_file = f"quiz_attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(quiz_log, f, indent=2)

print(f"Results saved to {output_file}")
print(f"\nNext step: python3 reform_agent.py {output_file}")