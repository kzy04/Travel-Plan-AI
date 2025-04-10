from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI model
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

# Initialize vector store
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="./travel_db", embedding_function=embedding_model)

# Load travel data into vector store
def load_travel_data(file_path, vector_store):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vector_store.add_documents(docs)

# Load travel data from file
travel_data_file = "data.txt"
load_travel_data(travel_data_file, vector_store)

# Define the output schema
class ItineraryItem(BaseModel):
    activity: str
    description: str
    time: str

class DayPlan(BaseModel):
    day: int
    itinerary: List[ItineraryItem]

class TransportationDetails(BaseModel):
    destinationTransport: str
    localTransport: List[str]

class TravelPlan(BaseModel):
    destination: str
    duration: int
    travelPlan: List[DayPlan]
    uniqueOptions: str
    safetyGuidelines: str
    transportationDetails: TransportationDetails

# Output parser
base_parser = PydanticOutputParser(pydantic_object=TravelPlan)
parser = OutputFixingParser.from_parser(parser=base_parser, llm=llm)

# Pydantic model to validate incoming request
class TravelQuestion(BaseModel):
    question: str
    answer: str

class TravelRequest(BaseModel):
    answers: List[TravelQuestion]

# Function to generate travel plan
def generate_rag_travel_plan(answers: Dict[str, str]) -> Dict[str, Any]:
    retriever = vector_store.as_retriever()
    query_text = " ".join([f"{q}: {a}" for q, a in answers.items()])
    retrieved_docs = retriever.invoke(query_text)

    user_preferences = "\n".join([f"{q}: {answers.get(q, 'Not provided')}" for q in answers])

    messages = [
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=f"""
            Based on the following preferences, generate a 3-day travel plan in Bangladesh.
            There can be multiple (activity, description, time) in each itinerary.
            Ensure the output strictly follows this JSON format schema.

            Preferences:
            {user_preferences}

            Additional travel information:
            {retrieved_docs}
        """),
    ]

    response = llm.invoke(messages)
    parsed_output = parser.parse(response.content)

    return parsed_output.dict()

# FastAPI endpoint to receive frontend requests
@app.post("/generate/")
async def generate_travel_plan(request: TravelRequest):
    # Convert list format into dictionary format
    answers_dict = {item.question: item.answer for item in request.answers}

    # Generate travel plan
    travel_plan = generate_rag_travel_plan(answers_dict)
    print(travel_plan)

    return travel_plan
