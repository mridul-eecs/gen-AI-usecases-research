import langgraph as lg
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine

# Set up MySQL Connection
DB_URI = "mysql+mysqlconnector://username:password@host/database"
engine = create_engine(DB_URI)
db = SQLDatabase(engine)

# Load FAQ Documents into Vector DB
faq_loader = TextLoader("faq_documents.txt")
faq_db = Chroma.from_documents(faq_loader.load(), embedding=OpenAIEmbeddings())

# Define the LLM
llm = Ollama(model="llama3")

# Define Query Execution Functions
def query_mysql(query):
    """Executes an SQL query on MySQL database."""
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error querying MySQL: {str(e)}"

def search_faq(query):
    """Searches the FAQ documents."""
    docs = faq_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

# Define Graph Nodes
def classify_query(state):
    """Determines which knowledge base to query."""
    user_query = state["query"].lower()
    if "database" in user_query or "sql" in user_query:
        return "mysql_query"
    elif "faq" in user_query or "help" in user_query:
        return "faq_search"
    return "fallback"

# Define Graph Workflow
workflow = lg.Graph()

workflow.add_node("mysql_query", query_mysql)
workflow.add_node("faq_search", search_faq)
workflow.add_node("fallback", lambda state: "I'm not sure how to answer that.")

# Define edges (logic flow)
workflow.add_edge("classify", "mysql_query", condition=lambda state: classify_query(state) == "mysql_query")
workflow.add_edge("classify", "faq_search", condition=lambda state: classify_query(state) == "faq_search")
workflow.add_edge("classify", "fallback")

# Set start node
workflow.set_entry_point("classify")

# Create LangGraph App
graph_app = lg.StatefulGraph(workflow)

# Run the Agent
user_input = {"query": "What is the latest transaction from the finance table?"}
response = graph_app.invoke(user_input)
print(response)
