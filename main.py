from typing import Union
from fastapi import FastAPI, Request, Response
import uvicorn
from fastapi.responses import JSONResponse
from nltk_utils import bag_of_words, tokenize, stem
from pydantic import BaseModel
import random
import json
import torch
import torch.nn as nn
from model import NeuralNet
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nltk
from fastapi.middleware.cors import CORSMiddleware

nltk.download('punkt')



app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestData(BaseModel):
    message: str

class ResponseData(BaseModel):
    response: str


@app.post("/chat", response_model=ResponseData)
async def chat(input: RequestData):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    msg = input.message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.50:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return {"response": random.choice(intent['responses'])}

    return {"response": "I do not understand..."}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port="PORT", default=5000, log_level="info")
