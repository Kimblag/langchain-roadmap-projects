# Python version of the notebook week3_langchain_intro\notebook.ipynb
import faiss
import numpy as np

from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline


hf_pipeline = pipeline(task="text2text-generation", model="google/flan-t5-large")
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# Read to verify it is ok
txt_path = "/content/sample_data/modelos_desarrollo_software.txt"
with open(txt_path, "r", encoding="utf-8") as f:
    contenido = f.read()
contenido[:500]

# 1. We use a txt file, so it is a Text-structured based and we need to divide this text in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_text(text=contenido)


# 2. Convert to Documents list
from langchain.docstore.document import Document
docs = [Document(page_content=chunk, metadata={"source": "RAG_doc"}) for chunk in chunks]


# 3. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([doc.page_content for doc in docs], show_profress_bar=True)


# 4. Save in FAISS
dim = embeddings.shape[1]
print(f"Embedding size: {embeddings.shape}, Dimension: {dim}")
index = faiss.IndexFlatL2(dim) # L2 = euclidian distance (menor distancia = más cercano / más parecido semánticamente)
index.add(np.array(embeddings, dtype="float32"))
print(f"Vectors quantity: {index.ntotal}")


"""## Find the most relevant chunk"""
def find_best_passages(query: str, index, top_k: int = 1):
  query_embedding = model.encode([query])
  distances, records_index = index.search(np.array(query_embedding, dtype="float32"), k=top_k)
  top_chunks = [chunks[i] for i in records_index[0]]
  return top_chunks


"""## Generate answer with LLM"""
def generate_answer_rag(query: str, best_passages: list, max_length: int = 256):
    context = "\n".join(best_passages)
    prompt = f"""
            Eres un asistente experto y conversacional. Usa el texto de referencia para dar tu respuesta,
            pero agrega explicaciones adicionales, ejemplos, y tu interpretación razonable para que sea más comprensible y natural.
            Pregunta: {query}
            Contexto: {context}
            Respuesta:
            """
    output = hf_pipeline(prompt, max_new_tokens=max_length)
    return output[0]['generated_text'].split("Respuesta:")[-1].strip()


query = "¿Qué es el Product Owner?"
best_passages = find_best_passages(query, index, top_k=7)
answer = generate_answer_rag(query, best_passages)
print(answer)