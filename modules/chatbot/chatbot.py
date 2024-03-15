from modules.chatbot.model import LLMHandler
from modules.chatbot.character import Character
from modules.chatbot.chatmemory import ChatMemory
import torch
import numpy as np


class ChatBot:
    def __init__(self, LLM: LLMHandler | str, character: Character | str = None, chat_memory: ChatMemory | str = None, **kwargs) -> None:
        self._change_model(LLM, *kwargs)
        self._change_character(character, *kwargs)
        self._change_memory(chat_memory, *kwargs)
        self.__system = self.LLM.encode('Me:', add_special_tokens = False)
        self.__user = self.LLM.encode('You:', add_special_tokens = False)

    def _change_model(self, model: LLMHandler | str, force_retokenize: bool = False, **kwargs):
        if isinstance(model, str):
            self.LLM = LLMHandler(model, *kwargs)
        else:
            self.LLM = model
        if hasattr(self, 'character') and ((self.character._tokenizer_name != self.LLM.model_name) or force_retokenize):
            self.character._retokenize(self.LLM.model_name, self.LLM.tokenizer, kwargs.get('add_special_tokens', True))
        if hasattr(self, 'memory') and ((self.memory._tokenizer_name != self.LLM.model_name) or force_retokenize):
            self.memory._retokenize(self.LLM.model_name, self.LLM.tokenizer, kwargs.get('add_special_tokens', True))

    def _change_character(self, character: Character | str = None, force_retokenize = False, **kwargs) -> None:
        if isinstance(character, str) or character is None:
            self.character = Character(character, None)
        else:
            self.character = character
        if (self.character._tokenizer_name != self.LLM.model_name) or force_retokenize:
            self.character._retokenize(self.LLM.model_name, self.LLM.tokenizer, kwargs.get('add_special_tokens', True))
    
    def _change_memory(self, memory: ChatMemory = None, force_retokenize = False, **kwargs) -> None:
        if memory is None:
            self.memory = ChatMemory(self.LLM.model_name)
        else:
            self.memory = memory
        if (self.memory._tokenizer_name != self.LLM.model_name) or force_retokenize:
            self.memory._retokenize(self.LLM.model_name, self.LLM.tokenizer, kwargs.get('add_special_tokens', True))
    
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
        prompt = prompt.cpu().numpy()
        responce = responce[prompt.shape[0]:]
        text_responce = self._procces_responce(responce, kwargs.get('skip_special_tokens', True))
        responce = self.LLM.encode(text_responce, add_special_tokens=False)
        self.memory.add_message(self.character.card['name'], text_responce, responce)
        return self.character.card['name'], text_responce
    
    def _procces_responce(self, responce: np.ndarray, skip_special_tokens = True) -> str:
        end = np.where(responce == self.__system[0])[0]
        if len(end) > 0:
            end = end[-1]
            responce = responce[:end]
        text_responce = self.LLM.decode(responce, skip_special_tokens=skip_special_tokens)
        end = text_responce.find('You:')
        if end != -1:
            text_responce = text_responce[:end]
        else:
            end = text_responce[::-1].find('.')
            if end != -1:
                text_responce = text_responce[:len(text_responce)-end]
        return text_responce