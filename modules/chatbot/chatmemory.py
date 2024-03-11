from torch import Tensor
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datetime import datetime

class ChatMemory:
    '''History of messages\n
    init:
         tokenizer_name: str = None --- name of model tokenizer'''
    
    def __init__(self, tokenizer_name: str = None) -> None:
        self.message_owner: list[str] = []
        self.history: list[str] = []
        self._tokenized_history: list[np.ndarray] = []
        self.datetime: list[str] = []
        self._tokenizer_name: str = tokenizer_name

    def add_message(self, owner: str, message: str, tokenized_message: Tensor | list | np.ndarray) -> None:
        '''add_message --- add a message to memory
        owner: str --- owner of the message (who wrote the text)
        message: str --- message
        tokenized_message: Tensor | list | np.ndarray --- encoded (tekenized) message'''
        self.message_owner.append(owner)
        self.history.append(message)
        match type(tokenized_message).__name__:
            case 'Tensor':
                tokenized_message = tokenized_message.numpy().astype(np.int32)
            case 'list':
                tokenized_message = np.array(tokenized_message, dtype=np.int32)
            case 'ndarray':
                pass
            case _:
                raise TypeError('tokenized_message must be Tensor, list or np.ndarray')
        self._tokenized_history.append(tokenized_message)
        self.datetime.append(str(datetime.now()))
    
    def _retokenize(self, tokenizer_name: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                    add_special_tokens: bool = True) -> None:
        '''_retokenize --- retokenize (reencode) all messages
        tokenizer_name: str --- a name of new tokenizer
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast --- new tokenizer
        add_special_tokens: bool = True --- adding special tokens (such as "<s>", "</s>" and "<unk>")'''
        self._tokenizer_name = tokenizer_name
        for i, text in enumerate(self.history):
            self._tokenized_history[i] = np.array(
                tokenizer.encode(text, add_special_tokens=add_special_tokens)).astype(np.int32)
        
    def get_history(self, message_number: int = 'all') -> list[str]:
        '''get_history --- get list of last messages
        message_number: int = 'all' --- number of last messages
        \nreturns a list of messages'''
        if message_number == 'all':
            return self.history
        return self.history[-message_number:]

    def get_tokenized_history(self, message_number: int = 'all') -> list[np.ndarray]:
        '''get_tokenized_history --- get list of last tokenized messages
        message_number: int = 'all' --- number of last messages
        \nreturns a list of tokenized messages'''
        if message_number == 'all':
            return self._tokenized_history
        return self._tokenized_history[-message_number:]
    
    def get_owners(self, message_number: int = 'all') -> list[str]:
        '''get_owners --- get list of last owners of messages
        message_number: int = 'all' --- number of last owners
        \nreturns a list of owners'''
        if message_number == 'all':
            return self.message_owner
        return self.message_owner[-message_number:]
    
    def clear_memory(self) ->  None:
        '''clear_memory --- delete all data (messages) from memory'''
        self.message_owner: list[str] = []
        self.history: list[str] = []
        self._tokenized_history: list[np.ndarray] = []
        self.datetime = list[str] = []