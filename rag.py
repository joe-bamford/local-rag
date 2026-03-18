import logging
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


model_name = "qwen3.5:4b"
n_chunks = 10

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
    search_kwargs={"k": n_chunks}  # fetch top n most relevant chunks
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

    Conversation history:
    {history}

    Question:
    {question}
    """
)

# --- BUILD RAG CHAIN ---
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def format_history(history):
    if not history:
        return "None"
    lines = []
    for msg in history:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "history": lambda x: format_history(x["history"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- CONVERSATION LOOP ---
chat_history = []

logging.info("RAG pipeline ready. Type 'exit' or 'quit' to end the session.")
logging.info(50*"-")

while True:
    query = input("\nYou: ").strip()
    if not query:
        continue
    if query.lower() in ("exit", "quit", "q"):
        break

    logging.info(f'Question: {query}')

    response = chain.invoke({"question": query, "history": chat_history})

    print(f"\nAssistant: {response}")
    logging.info(f'Response: {response}')

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))