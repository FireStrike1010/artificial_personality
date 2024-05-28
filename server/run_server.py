from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field

import torch
from modules.modelhandler.core import LLM
from modules.modelhandler.chatbot import ChatBot

from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()
model = os.getenv('model', None)
if model is None:
    raise Exception('Specify model in .env file')
device = os.getenv('device', 'cuda')
do_sample = os.getenv('do_sample', 'True').lower()
if do_sample in ('1', 'true', 't', 'y', 'yes'):
    do_sample = True
else:
    do_sample = False
temperature = float(os.getenv('temperature', 0.8))
top_p = float(os.getenv('top_p', 0.8))
top_k = int(os.getenv('top_k', 40))
max_new_tokens = int(os.getenv('max_new_tokens', 64))
template = os.getenv('template', None)
add_start = os.getenv('add_start', 'True').lower()
if add_start in ('1', 'true', 'true', 't', 'y', 'yes'):
    add_start = True
else:
    add_start = False

model = LLM(model_path=model, device=device)
chatbot = ChatBot(model)

class GenerateQuery(BaseModel):
    prompt: Optional[str] = Field(None, description="The prompt for generating the query")
    history: Optional[list[dict[str, str]]] = Field(None, description="History of interactions")
    context: Optional[list[dict[str, str]] | str] = Field(None, description="Context for query generation")
    do_sample: bool = Field(do_sample, description="Whether to sample the output")
    temperature: float = Field(temperature, description="Temperature sampling temperature")
    top_p: float = Field(top_p, description="Top-p sampling parameter")
    top_k: int = Field(top_k, description="Top-k sampling parameter")
    max_new_tokens: int = Field(max_new_tokens, description="Maximum number of new tokens to generate")
    add_start: bool = Field(add_start, description="Whether to add a start token")
    chat_template: Optional[str] = Field(None, description="Chat template for query generation")

app = FastAPI()

@app.post('/query')
async def process_generate_query(query: GenerateQuery) -> dict[str, dict[str, str]]:
    global chatbot
    return chatbot(**dict(query))

host = os.getenv('host', '127.0.0.1')
port = int(os.getenv('port', 8000))
ping_timeout = int(os.getenv('ping_timeout', 1000))
restart = os.getenv('reload', 'True')
if restart in ('1', 'true', 't', 'y', 'yes'):
    restart = True
else:
    restart = False
restart_delay = float(os.getenv('restart_delay', 0.25))


if __name__ == '__main__':
    uvicorn.run(app, host=host, port=port, ws_ping_timeout=ping_timeout, reload=restart, reload_delay=restart_delay)
