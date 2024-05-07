from fastapi import security, FastAPI
from pydantic import BaseModel
import asyncio
import json
from typing import Optional
from .chatbot import ChatBot
from .chatbot import LLM

class Query(BaseModel):
    prompt: str | None = None
    history: list[dict[str, str]] | None = None
    context: list[dict[str, str]] | str | None = None
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.2
    top_k: int = 40
    max_new_tokens: int = 64
    add_start: bool = True
    chat_template: str | None = None

app = FastAPI()
model = LLM('C:/Users/FireStrike/Documents/project2024/text2text/models/TheBloke_Yarn-Mistral-7B-128k-AWQ')
chatbot = ChatBot(model)

@app.get('/query')
async def process_query(data = Query()):
    print(data)
    data = json.loads(data) #type: ignore
    ans = chatbot(data.get('prompt'), data.get('history'), data.get('context'), do_sample=data.get('do_sample'),
                  temperature=data.get('temperature'), top_p=data.get('top_p'),
                  top_k=data.get('top_k'), max_new_tokens=data.get('max_new_tokens'),
                  add_start=data.get('add_start'), chat_template=data.get('chat_template'))
    return ans
    
