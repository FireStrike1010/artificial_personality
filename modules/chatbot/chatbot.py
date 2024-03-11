from modules.chatbot.model import LLMHandler
from modules.chatbot.character import Character
from modules.chatbot.chatmemory import ChatMemory
import torch
import numpy as np
from pathlib import Path


class ChatBot:
    def __init__(self, LLM: LLMHandler | str, character: Character = None, chat_memory: ChatMemory = None,
                 check_tokenized: bool = True, **kwargs) -> None:
        if isinstance(LLM, str):
            self.LLM = LLMHandler(LLM)
        else:
            self.LLM = LLM
        self.memory = None
        self.character = None
        if (character is None) and (self.memory is None):
            self.character = Character(None, self.LLM.model_name)
            self.character._retokenize(self.LLM.model_name, self.LLM.tokenizer, kwargs.get('add_special_tokens', True))
        else:
            self.character = character
        self.memory = ChatMemory(self.LLM.model_name) if (chat_memory is None) and (self.memory is None) else chat_memory
        self.username = 'User'
        self.__system = np.array(self.LLM.encode('Me:', add_special_tokens = False)).astype(np.int32)
        self.__user = np.array(self.LLM.encode('You:', add_special_tokens=False)).astype(np.int32)
        if check_tokenized:
            self.check_tokenized(kwargs.get('add_special_tokens', True), kwargs.get('force_retokenize', False))

    def check_tokenized(self, add_special_tokens: bool = True, force_retokenize: bool = False) -> None:
        if force_retokenize or (self.memory._tokenizer_name != self.LLM.model_name):
            self.memory = self.memory._retokenize(self.LLM.model_name, self.LLM.tokenizer,
                                                  add_special_tokens=add_special_tokens)
        if force_retokenize or (self.character._tokenizer_name != self.LLM.model_name):
            self.character._retokenize(self.LLM.model_name, self.LLM.tokenizer, add_special_tokens=add_special_tokens)
    
    def start(self, username: str = 'User') -> tuple[str, str]:
        self.username = username
        self.memory.add_message(self.character.card['name'],
                                self.character.card['greeting'],
                                self.LLM.encode(self.character.card['greeting']))
        return self.character.card['name'], self.character.card['greeting']
    
    def _create_prompt(self, use_character: bool = True, use_memory: bool = True,
                       message_number: int = 'all', add_start_owner: bool = True) -> torch.Tensor:
        prompt = []
        if use_character:
            prompt.append(self.character._tokenized_context)
        if use_memory:
            for i in self.memory.get_tokenized_history(message_number):
                prompt.append(i)
        if add_start_owner:
            prompt.append(np.array([1]))
            prompt.append(self.__system)
        prompt = np.hstack(prompt)
        prompt = torch.Tensor(prompt).type(torch.int32).cuda(self.LLM._device)
        return prompt

    def send_message(self, message: str, **kwargs) -> tuple[str, str]:
        tokenized_message = self.LLM.encode(message, kwargs.get('add_special_tokens', True))
        tokenized_message = np.hstack((self.__user, tokenized_message))
        self.memory.add_message(self.username, message, tokenized_message)
        return self.username, message
    
    def get_responce(self, **kwargs) -> tuple[str, str]:
        prompt = self._create_prompt(kwargs.get('use_character', True), kwargs.get('use_memory', True),
                                     kwargs.get('message_number', 'all'), kwargs.get('add_start_owner', True))
        responce = self.LLM.generate(prompt, *kwargs)
        text_responce = self._procces_responce(
            responce, kwargs.get('max_new_tokens', self.LLM._default_generate_params['max_new_tokens']))
        self.memory.add_message(self.character.card['name'], text_responce, responce)
        return self.character.card['name'], text_responce
    
    def _procces_responce(self, responce: torch.Tensor, max_new_tokens: int, skip_special_tokens = True) -> str:
        responce = responce[-max_new_tokens:]
        responce = self.LLM.decode(responce, skip_special_tokens=skip_special_tokens)
        end = responce.find('You:')
        if end != -1:
            end -= 1
            responce = responce[:end]
        return responce