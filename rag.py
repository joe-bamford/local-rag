import logging
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


model_name = "qwen3.5:9b"

# --- LOAD CONTEXT FILE ---
with open("canterbury.txt", "r") as f:
    raw_text = f.read()

# --- SPLIT & CHUNK CONTEXT ---

# Recursive splitter (general-purpose)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # characters per chunk
    chunk_overlap=200,     # overlap between chunks to avoid losing context at boundaries
)
chunks = splitter.create_documents([raw_text])

# # Markdown splitter (for structured text with headers)
# splitter = MarkdownHeaderTextSplitter(
#     headers_to_split_on=[
#         ("#", "title"),
#         ("##", "section"),
#         ("###", "subsection"),
#     ],
#     # Ensure headers are included in chunks
#     strip_headers=False 
# )
# chunks = splitter.split_text(raw_text)

# --- EMBED & STORE IN VECTOR DB ---
persist_directory = "./chroma_db"

logging.info("Creating ChromaDB...")
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,  # save to disk
)

# --- SET UP RETRIEVER ---
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}  # fetch top N most relevant chunks
)

# --- SET UP LLM & PROMPT ---
llm = OllamaLLM(model=model_name)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the context below by carefully reviewing all provided context chunks.
    If information spans multiple chunks, synthesize them into a complete answer.
    The context is the General Prologue of the Canterbury Tales by Chaucer.
    If the answer isn't in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)

# --- BUILD RAG CHAIN ---
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- QUERY ---

# PASSING
# query = "How many pilgrims are there in the story?"
# query = "In what time of year does the story start?"
# query = "What is the pilgrims' destination?"
# query = "What does the Merchant wear on his head?"

# FAILING
# query = "In which battles has the Knight fought?"
query = "List all of the characters described in the order they appear."
# query = "What is the Monk wearing?"


logging.info(50*"-")
logging.info(f'Question: {query}')
logging.info(50*"-")

response = chain.invoke(query)

logging.info(50*"-")
logging.info(f'Response: {response}')
logging.info(50*"-")