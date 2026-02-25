from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation'
)

llm = ChatHuggingFace(llm=llm_endpoint)

loader = PyPDFLoader("../rag/fact_sheet_22_diabetic_retinopathy_new.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant info from diabetic retinopathy PDF."""

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    return f"""
Use the following context to answer the question.
You are a Retinopathy medical assistant that helps answer questions about diabetic retinopathy based on the provided context. If the question cannot be answered using the context, say you don't know.
Context:
{context}

Question:
{query}
"""


tools = [rag_tool]
llm_tool = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _clean_messages(messages):
    """Filter messages to only keep Human/AI content messages.
    Removes ToolMessages and strips tool_calls from AIMessages
    so HuggingFace API doesn't reject them."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        cleaned.append(msg)
    return cleaned


def chat_node(state: ChatState):
    messages = _clean_messages(state['messages'])
    response = llm_tool.invoke(messages)
    return {'messages': [response]}


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node('chat node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat node')
graph.add_conditional_edges('chat node', tools_condition)
graph.add_edge('tools', 'chat node')

memory = MemorySaver()
app = graph.compile(checkpointer=memory)



