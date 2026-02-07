import chromadb
import ollama

try:
    from config import DEFAULT_MODEL
except ImportError:
    DEFAULT_MODEL = "llama3:8b"

# Load the database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("unit_materials")

def search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

def extract_search_terms(question, model=None):
    """Ask the LLM to identify key concepts to search for"""
    model = model or DEFAULT_MODEL
    response = ollama.chat(model=model, messages=[
        {'role': 'user', 'content': f'What are the key operations management concepts in this question? List only the technical terms, separated by commas. No explanation.\n\nQuestion: {question}'}
    ])
    return response['message']['content']

def answer_question(question, model=None):
    model = model or DEFAULT_MODEL
    search_terms = extract_search_terms(question, model=model)
    chunks = search(search_terms, n_results=3)
    context = "\n\n---\n\n".join(chunks)

    response = ollama.chat(model=model, messages=[
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

# Test with the utilisation question
question = """
Suppose a cake shop has a design capacity of 8.5 cakes per day and an effective capacity 1.9 cakes per day. Yesterday, the cake shop produced 8.5 cakes. What is the utilisation of the cake shop as a percentage?

Options:
a. 805%
b. 20.0%
c. 100%
d. 100%
e. 155%
f. 2.24%
g. 22.4%
h. 100%
"""

result = answer_question(question)

print("RETRIEVED CONTEXT:")
print("="*50)
for i, chunk in enumerate(result['retrieved_context']):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:300] + "...")

print("\n\nANSWER:")
print("="*50)
print(result['answer'])