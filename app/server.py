from fastapi import FastAPI  # For building the API
from langchain.prompts import ChatPromptTemplate  # For creating chat prompts
from langchain_openai import ChatOpenAI  # For OpenAI chat functionality
from langchain_groq import ChatGroq  # For Groq chat functionality
from langchain_community.llms import Ollama  # For Ollama LLM
from langserve import add_routes  # For adding routes to the API
import uvicorn  # For running the API
import os  # For loading environment variables



app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)


# llm=Ollama(model="llama3")
llm = 'taras_model'
prompt = ChatPromptTemplate.from_template("Напиши вірш на тему {topic} на 50 слів, пж")


add_routes(
    app,
    prompt | llm, 
    path = "/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

