from datetime import datetime
from typing import Optional

class ChatMemory:
    '''History of messages'''
    def __init__(self) -> None:
        '''Create a ChatMemory object\n'''
        self.history: list[dict[str, str]] = []
        self.datetime: list[str] = []

    def add_message(self, owner: Optional[str] = None, message: Optional[str] = None) -> None:
        '''Add a message to memory\n
        args:
                owner: str --- owner of the message
                message: str --- message in string format'''
        if owner is None or message is None:
            return
        self.history.append({'role': owner, 'content': message})
        self.datetime.append(str(datetime.now()))
        
    def get_history(self, message_number: int | str = 'all') -> list[dict[str, str]]:
        '''Get list of last messages
        args:
            message_number: int = 'all' --- number of last messages
        \nreturns a list of messages'''
        if isinstance(message_number, str):
            return self.history
        return self.history[-message_number:]
    
    def clear_memory(self) ->  None:
        '''Delete all data (messages) from memory'''
        self.history: list[dict[str, str]] = []
        self.datetime: list[str] = []