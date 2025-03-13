# INSTALL REQUIRED PACKAGES (if not installed)
# pip install openai langchain langchain-community langchain-openai langchain-chroma chromadb streamlit python-dotenv

# LOADING API KEY
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("API Key loaded:", openai_api_key[:10] + "********")  

# IMPORTING REQUIRED MODULES
from langchain_openai  import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# INITIALIZE OPENAI MODEL
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

# FUNCTION TO LOAD TRAVEL DATA FROM FILE INTO VECTOR DATABASE
def load_travel_data(file_path, vector_store):
    """Reads the text file and adds its content to the vector database."""
    
    # Load travel data from text file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split large text into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Add processed text data to the vector store
    vector_store.add_documents(docs)
    print("Travel data successfully added to the vector store!")

# INITIALIZE VECTOR STORE
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="./travel_db", embedding_function=embedding_model)

# LOAD TRAVEL DATA FROM FILE (Replace 'travel_data.txt' with your actual file)
travel_data_file = "travel_data.txt"  
load_travel_data(travel_data_file, vector_store)

# FUNCTION TO GENERATE A TRAVEL PLAN USING RAG
def generate_rag_travel_plan(destination, interests, duration):
    """Generates a travel plan by retrieving relevant data from the vector store."""
    
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(destination)

    messages = [
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=f"I want to visit {destination} for {duration} days. "
                             f"My interests are {interests}. Can you suggest a detailed travel plan?"),
        SystemMessage(content=f"Here is some additional travel information: {retrieved_docs}")
    ]
    
    return llm.invoke(messages).content

# TEST THE TRAVEL PLAN GENERATOR
print(generate_rag_travel_plan("Dhaka", "Food, museums, malls", 2))
