# Import necessary modules
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Retrieve API keys from environment variables
openai_api_key = os.environ["OPENAI_API_KEY"]  # OpenAI API key
tavily_api_key = os.environ["TAVILY_API_KEY"]  # Tavily API key

# Define the Search tools
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize the Tavily search tool
search_tool = TavilySearchResults()

# Use the search tool to query for specific information
#result = search_tool.invoke("What was the score of the Real Madrid - Barcelona match played yesterday?")

# Print the result of the query
#print(result)

# The above code initializes a search tool using the Tavily API and queries it for information about a specific football match.

# RAG Tool loading data
# Import necessary modules for document loading and vector storage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents from a web page
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Split the loaded documents into smaller chunks for processing
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

# Create a FAISS vector store from the documents using OpenAI embeddings
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

# Use the retriever to find relevant documents based on a query
retriever.get_relevant_documents("how to upload a dataset")[0]

# The above code loads documents from a specified web page, splits them into smaller chunks, and creates a FAISS vector store for efficient retrieval of relevant documents based on a query.
# RAG Tool loading data from a local file
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Use the retriever tool to query for specific information
#result = retriever_tool.invoke("How to upload a dataset?")
# Print the result of the query 
#print(result)
# The above code creates a retriever tool for searching information about LangSmith and queries it for specific information about uploading a dataset.
# RAG Tool loading data from a local file

# Create custom tools
# Import necessary modules for creating custom tools
from langchain.agents import tool
@tool
def get_word_length_tool(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

get_word_length_tool.invoke("julio")
# The above code defines a custom tool that returns the length of a given word and invokes it with the word "julio".

#Create list of tools available to the agent
tools = [search_tool, retriever_tool, get_word_length_tool]

# Create the agent
# Choose the llm model to use for the agent

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
from langchain import hub

# Get the prompt to use - you can modify this!
# Define the agent prompt that will guide the agent's behavior

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

# Initialize the agent with the llm, tools and prompt
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

# Given our input , the agent will decide the actions to take
# The agent will not take any action, agent executor will take care of that

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Run the agent executor with a specific input
#agent_executor.invoke({"input": "hi! who are you?"})
agent_executor.invoke({"input": "in less than 20 words, what is langsmith?"})
agent_executor.invoke({"input": "Who scored the last goal in the Real Madrid - Barcelona soccer match yesterday?"})
agent_executor.invoke({"input": "How many letters are in the word AIAccelera?"})



