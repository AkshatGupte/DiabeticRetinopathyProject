import redis
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from dotenv import load_dotenv
import os

load_dotenv()

# Embeddings + Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = ChatGroq(model="openai/gpt-oss-20b")

# Pinecone vector store
index_name = "retinopathy"
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Format retrieved documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# RAG Inputs
rag_chain_inputs = RunnableParallel(
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough()
    }
)

prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template="""
You are a medical assistant specializing in diabetic retinopathy.
Use ONLY the following context to answer the question.
If the query is irrelevant to diabetic retinopathy, respond with "I don't know."
If the answer isn't in the context, reply: "I don't know."

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:
"""
)

# Final RAG Chain
rag_chain = (
    rag_chain_inputs
    | RunnableLambda(lambda x: {
        "context": x["context"],
        "question": x["question"],
        "history": x.get("history", "")
    })
    | RunnableLambda(lambda x: prompt.format(**x))   # <--- FIXED LINE
    | RunnableLambda(lambda text: model.invoke([{"role": "user", "content": text}]))
    | RunnableLambda(lambda msg: msg.content)
)



# Redis Chat History
redis_client = redis.Redis(host="localhost", port=6379)

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        redis_client=redis_client
    )

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_redis_history,
    input_messages_key="question",
    history_messages_key="history",
)
