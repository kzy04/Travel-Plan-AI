#pip install openai langchain langchain-community chromadb streamlit python-dotenv
#pip install -U openai langchain langchain-community langchain-openai langchain-chroma chromadb streamlit python-dotenv

# LOADING API_KEY
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
# RAG IMPLEMENTATION
from langchain_chroma  import Chroma
from langchain_openai  import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("API Key loaded:", openai_api_key[:10] + "********")  


# CREATING A SIMPLE TRAVEL PLAN GENERATOR
from langchain_openai  import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Initialize the OpenAI model
#llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")



# Example Usage
destination = "Paris"
interests = "history, museums, food"
duration = 2

#travel_plan = generate_travel_plan(destination, interests, duration)
#print(travel_plan)


# FUNCTION TO LOAD TRAVEL DATA FROM FILE INTO VECTOR DATABASE
def load_travel_data(file_path, vector_store):
    """Reads the text file and adds its content to the vector database."""
    
    # Load travel data from text file
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Print to verify correct data is being loaded
    print("Loaded Documents:", documents)

    # Split large text into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Add processed text data to the vector store
    vector_store.add_documents(docs)
    print("Travel data successfully added to the vector store!")

# Initialize vector store and embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="./travel_db", embedding_function=embedding_model)

# Reset vector store by deleting old data
vector_store.delete_collection()  # Clears the database
vector_store = Chroma(persist_directory="./travel_db", embedding_function=embedding_model)  # Reinitialize

# LOAD TRAVEL DATA FROM FILE (Replace 'travel_data.txt' with your actual file)
travel_data_file = "data.txt"  
load_travel_data(travel_data_file, vector_store)


# RETRIEVING RELEVENT DATA
from langchain.chains import RetrievalQA



def retrieve_travel_info(destination):
    """Retrieves relevant travel information from the stored vector database."""
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(destination)
    
    if retrieved_docs:
        return "\n".join([doc.page_content for doc in retrieved_docs])
    else:
        return f"Sorry, no travel information found for {destination}."

# TEST THE RETRIEVAL FUNCTION (Only RAG Output)
print(retrieve_travel_info("Paris food"))


