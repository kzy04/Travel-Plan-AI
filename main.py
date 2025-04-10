from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Loading travel data into vector store
def load_multiple_texts(data_folder: str, vector_store):
    files = glob.glob(f"{data_folder}/*.txt")

    for file_path in files:
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        vector_store.add_documents(docs)

# Loading the folder
load_multiple_texts("travel_data", vector_store)

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
            Ensure the output strictly follows this JSON format:
            ```
            {{
              "destination": "Bangladesh",
              "duration": 3,
              "travelPlan": [
                {{ "day": 1, "itinerary": [ {{ "activity": "", "description": "", "time": "" }} ] }},
                {{ "day": 2, "itinerary": [ {{ "activity": "", "description": "", "time": "" }} ] }},
                {{ "day": 3, "itinerary": [ {{ "activity": "", "description": "", "time": "" }} ] }}
              ],
              "uniqueOptions": "",
              "safetyGuidelines": "",
              "transportationDetails": {{
                "destinationTransport": "",
                "localTransport": ["", "", ""]
              }}
            }}
            ```
            Only return valid JSON. Do not include extra text or explanations.
            
            Preferences:
            {user_preferences}
            
            Additional travel information:
            {retrieved_docs}
        """),
    ]

    response = llm.invoke(messages).content

    # Ensure valid JSON output
    try:
        itinerary_json = json.loads(response)
        return itinerary_json
    except json.JSONDecodeError:
        print ("Invalid JSON detected. Retrying with stricter enforcement...")
        
        # Second attempt with a more structured re-prompt
        messages.append(HumanMessage(content="Your last response was not valid JSON. Please strictly return only the JSON structure as specified."))

        response = llm.invoke(messages).content

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print ("Invalid JSON detected. Retrying with stricter enforcement...")

            # Third attempt with a more structured re-prompt
            messages.append(HumanMessage(content="Your last response was not valid JSON. Please strictly return only the JSON structure as specified."))

            response = llm.invoke(messages).content

            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print ("Invalid JSON detected. Retrying with stricter enforcement...")

                # Fourth attempt with a more structured re-prompt
                messages.append(HumanMessage(content="Your last response was not valid JSON. Please strictly return only the JSON structure as specified."))

                response = llm.invoke(messages).content

                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    print ("Invalid JSON detected. Retrying with stricter enforcement...")

                    # Fifth attempt with a more structured re-prompt
                    messages.append(HumanMessage(content="Your last response was not valid JSON. Please strictly return only the JSON structure as specified."))

                    response = llm.invoke(messages).content

                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response from the model after retrying 5 times. Try refining the prompt."}


# FastAPI endpoint to receive frontend requests
@app.post("/generate/")
async def generate_travel_plan(request: TravelRequest):
    # Convert list format into dictionary format
    answers_dict = {item.question: item.answer for item in request.answers}

    # Generate travel plan
    travel_plan = generate_rag_travel_plan(answers_dict)
    print(travel_plan)

    return travel_plan
