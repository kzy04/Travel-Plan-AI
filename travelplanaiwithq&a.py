# INSTALL REQUIRED PACKAGES (if not installed)
# pip install openai langchain langchain-community langchain-openai langchain-chroma chromadb streamlit python-dotenv

# LOADING API KEY
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#print("API Key loaded:", openai_api_key[:10] + "********")  

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
    #print("Travel data successfully added to the vector store!")

# INITIALIZE VECTOR STORE
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="./travel_db", embedding_function=embedding_model)

# LOAD TRAVEL DATA FROM FILE (Replace 'travel_data.txt' with your actual file)
travel_data_file = "data.txt"  
load_travel_data(travel_data_file, vector_store)

# FUNCTION TO GENERATE A TRAVEL PLAN USING RAG
def generate_rag_travel_plan(answers):
    """Generates a travel plan by retrieving relevant data from the vector store."""
    
    retriever = vector_store.as_retriever()
    query_text = " ".join([f"{q}: {a}" for q, a in answers.items()])
    retrieved_docs = retriever.invoke(query_text)

    questions = [
        "What type of travel experiences interest you most?",
        "What travel pace do you prefer?",
        "Will you be traveling alone or with others?",
        "Are there any past travel experiences you particularly loved or disliked?",
        "Which budget best describes your trip?",
        "Do you have a specific daily spending limit?",
        "Do you have any mobility or accessibility requirements?",
        "Do you have any dietary restrictions or allergies?",
        "Which language(s) do you prefer?",
        "How do you prefer to travel to and from your destination?",
        "How do you plan to get around once you’re there?",
        "How important is food in your travel experience?",
        "What kind of dining experiences do you enjoy?",
        "What is your comfort level with exploring less typical tourist destinations?",
        "Are there any specific countries or regions you want to visit or avoid?",
        "Are there any specific safety concerns you want to address?",
        "How do you prefer your accommodation to be located?",
        "When do you plan to travel?",
        "How flexible are your travel dates?",
        "Are eco-friendly or sustainable options important to you?",
        "What type of accommodation do you prefer?",
        "Are you interested in unique or unconventional accommodations?",
        "Are there any medical conditions or health concerns we should consider?",
        "Is there anything else we should know to plan your trip?"
    ]
    
    user_preferences = "\n".join([f"{q}: {answers.get(q, 'Not provided')}" for q in questions])
    
    messages = [
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=f"Based on the following preferences, generate a 3 day detailed travel plan in Bangladesh, strictly in json format: \n{user_preferences}"),
        SystemMessage(content=f"Here is some additional travel information: {retrieved_docs}")
    ]
    
    response = llm.invoke(messages).content

    # Ensure valid JSON output
    try:
        itinerary_json = json.loads(response)
        return itinerary_json  # Return JSON object
    except json.JSONDecodeError:
        return {"error": "The model returned invalid JSON. Please refine the prompt or check response formatting."}

# TEST THE TRAVEL PLAN GENERATOR
sample_answers = {
    "What type of travel experiences interest you most?": "Cultural, food, and historical sightseeing",
    "What travel pace do you prefer?": "Moderate",
    "Will you be traveling alone or with others?": "With a partner",
    "Are there any past travel experiences you particularly loved or disliked?": "Loved exploring ancient ruins in Italy; disliked the feeling of being rushed on a group tour.",
    "Which budget best describes your trip?": "Mid-range",
    "Do you have a specific daily spending limit?": "Approximately $200 per day",
    "Do you have any mobility or accessibility requirements?": "No",
    "Do you have any dietary restrictions or allergies?": "Lactose intolerant",
    "Which language(s) do you prefer?": ["English", "Basic Spanish"],
    "How do you prefer to travel to and from your destination?": "Flights",
    "How do you plan to get around once you’re there?": "Public transportation, walking, and occasional taxis",
    "How important is food in your travel experience?": "Very important",
    "What kind of dining experiences do you enjoy?": "Local restaurants, cafes, and food tours",
    "What is your comfort level with exploring less typical tourist destinations?": "High",
    "Are there any specific countries or regions you want to visit or avoid?": "Interested in visiting Spain and Portugal; prefer to avoid extremely hot climates during peak summer.",
    "Are there any specific safety concerns you want to address?": "Prefer well-lit areas at night and secure accommodations.",
    "How do you prefer your accommodation to be located?": "Within walking distance of attractions or public transportation.",
    "When do you plan to travel?": "September 2024",
    "How flexible are your travel dates?": "Very flexible",
    "Are eco-friendly or sustainable options important to you?": "Yes, somewhat important",
    "What type of accommodation do you prefer?": "Boutique hotels, apartments, or guesthouses",
    "Are you interested in unique or unconventional accommodations?": "Yes, if they offer a unique cultural experience",
    "Are there any medical conditions or health concerns we should consider?": "No",
    "Is there anything else we should know to plan your trip?": "Interested in attending local festivals or events if possible."
}

travel_plan_json = generate_rag_travel_plan(sample_answers)
print(json.dumps(travel_plan_json, indent=2))
