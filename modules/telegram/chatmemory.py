
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datetime import datetime
from pathlib import Path

class ChatMemory:
    '''History of messages'''
    def __init__(self, username: str | None = None) -> None:
        self.history: list[dict[str, str]] = []
        self.datetime: list[str] = []
        self._username = username

    def add_message(self, owner: str, message: str) -> None:
        '''add_message --- add a message to memory
        owner: str --- owner of the message (who wrote that text)
        message: str --- message'''
        self.history.append({'role': owner, 'content': message})
        self.datetime.append(str(datetime.now()))
        
    def get_history(self, message_number: int | str = 'all') -> list[dict[str, str]]:
        '''get_history --- get list of last messages
        message_number: int = 'all' --- number of last messages
        \nreturns a list of messages'''
        if isinstance(message_number, str):
            return self.history
        return self.history[-message_number:]
    
    def clear_memory(self) ->  None:
        '''clear_memory --- delete all data (messages) from memory'''
        self.history: list[dict[str, str]] = []
        self.datetime: list[str] = []