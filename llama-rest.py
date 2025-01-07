from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import faiss
import numpy as np
import os

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")
gen_model = AutoModelForCausalLM.from_pretrained("ollama/ollama3")
index = faiss.IndexFlatL2(768)

def load_and_embed_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load_and_split())

    embeddings = []
    for doc in documents:
        text = doc.page_content
        inputs = tokenizer(text, return_tensors='pt')
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)
    index.add(embeddings)
    return documents

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data['query']

    inputs = tokenizer(query_text, return_tensors='pt')
    outputs = embedding_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    D, I = index.search(query_embedding, k=5)

    context = " ".join([documents[i].page_content for i in I[0]])

    prompt = f"Context: {context}\nQuestion: {query_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = gen_model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response})

@app.route('/load_pdfs', methods=['POST'])
def load_pdfs():
    data = request.json
    pdf_paths = data['pdf_paths']
    global documents
    documents = load_and_embed_pdfs(pdf_paths)
    return jsonify({'status': 'PDFs loaded and indexed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
