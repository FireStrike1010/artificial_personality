from .core import LLM
import torch
import numpy as np
import os
import pathlib
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteriaList
from typing import Optional, Iterable


class ChatBot:
    '''ChatBot that handles messages and creates chat templates'''

    def __init__(self, llm: LLM, **kwargs) -> None:
        '''Create a ChatBot object\n
        args:
                llm: LLM --- Large Language model object
        kwargs:
                chat_template: Optional[str | os.Pathlike] = None --- .jinja template that describes
                how to handle a messages and context, if None - autosearch in model folder (even if this file isn't included the model would handle messages by defult custom template)
                (see more how they work)
                https://huggingface.co/docs/transformers/main/en/chat_templating
                my_name: str = '<Me>' --- how the model the model identifies itself
                (make sure to wrap with some symbols a word or name)
                user_name: str = '<You>' --- how the model identifies user
                (make sure to wrap with some symbols a word or name)
                system_name: str = '<System>' --- how the model identifies a character context
                (make sure to wrap with some symbols a word or name)
                stop_words: list[str] = [] --- words that stops generation
                (my_name, user_name, system_name are already included, this is necessary to prevent continuing conversation generating)'''
        def get_template_path(folder_path: os.PathLike | str) -> os.PathLike | None:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file = pathlib.Path(file)
                    if file.suffix == '.jinja':
                        return pathlib.Path(root, file)
        self.llm = llm
        self.chat_template = kwargs.get('chat_template', self.llm._model_path)
        if isinstance(self.chat_template, str) and '.jinja' not in self.chat_template:
            self.chat_template = self.chat_template
        else:
            self.chat_template = pathlib.Path(self.chat_template)
            if not self.chat_template.is_file():
                self.chat_template = get_template_path(self.chat_template)
                if self.chat_template is not None:
                    with open(self.chat_template, 'r') as file:
                        self.chat_template = file.read()
            else:
                with open(self.chat_template, 'r') as file:
                    self.chat_template = file.read()
        if self.chat_template is not None:
            self.chat_template = self.chat_template.replace('\n', '').replace('  ', '')
        self.my_name: str = kwargs.get('my_name', '<Me>')
        self.user_name: str = kwargs.get('user_name', '<You>')
        self.system_name: str = kwargs.get('system_name', '<System>')
        stop_words: list = [self.my_name, self.user_name, self.system_name]
        stop_words += kwargs.get('stop_words', [])
        self.stoping_criteria: StoppingCriteriaList = StoppingCriteriaList([StoppingCriteria(self.llm._device, self.llm.tokenizer, stop_words)])

    def __call__(self, prompt: Optional[str] = None,
                 history: Optional[list[dict[str, str]]] = None,
                 context: Optional[list[dict[str, str]] | str] = None, **kwargs) -> dict[str, dict[str, str]]:
        '''Generate a responce of the model\n
        args:
                prompt: Optional[str] = None --- new message (request)
                history: Optional[list[dict[str, str]]] = None --- chatting history [{'role': '<Me>', 'content': 'Hello, how are you?'}, ...] or something like that
                context: Optional[list[dict[str, str] | str]] = None --- could be an example of charater chatting or describing factors
        kwargs: --- default settings for generating
                do_sample: bool = True
                stopping_criteria: Optional[StoppingCriteriaList | list[StoppingCriteria]] = None
                temperature: float = 0.8
                top_p: float = 0.2
                top_k: int = 40
                max_new_tokens: int = 64
                add_start: bool = True --- adding bos for model generation
                chat_template: str = self.chat_template
        /nReturns a dict[str, dict[str, str]] --- {'input': {'role': '<You>', 'content': 'input_message(prompt)'}, 'output': {'role': '<Me>', 'generated_message'}}'''
        add_start = kwargs.get('add_start', True)
        chat_template = kwargs.get('chat_template')
        result: dict[str, dict[str, str]] = {'input': {'role': self.user_name,
                                                       'content': prompt.strip() if prompt else ''}}
        inp = self._apply_chat_template(prompt, history, context, chat_template, add_start)
        inp_tokenized = self.llm.encode(inp, False, 'Tensor')
        generate_params = {'do_sample': kwargs.get('do_sample', True),
                           'stopping_criteria': kwargs.get('stopping_criteria', self.stoping_criteria),
                           'temperature': kwargs.get('temperature', 0.8),
                           'top_p': kwargs.get('top_p', 0.2),
                           'top_k': kwargs.get('top_k', 40),
                           'max_new_tokens': kwargs.get('max_new_tokens', 64)}
        out_tokenized = self.llm.generate(inp_tokenized, **generate_params)
        out_tokenized = out_tokenized[inp_tokenized.shape[1]:] #type: ignore
        for stop in self.stoping_criteria[0].stops:
            if torch.all(out_tokenized[-stop.shape[0]:] == stop):
                out_tokenized = out_tokenized[:-stop.shape[0]]
                break
        out_tokenized = out_tokenized.cpu().numpy()
        out = self.llm.decode(out_tokenized, skip_special_tokens=kwargs.get('skip_special_tokens', True))
        result['output'] = {'role': self.my_name, 'content': out.strip()}
        return result
    
    def _apply_chat_template(self, prompt: Optional[str] = None,
                             history: Optional[list[dict]] = None,
                             context: Optional[list[dict] | str] = None,
                             chat_template: Optional[str] = None,
                             add_start: bool = True) -> str:
        '''Applies a template for generating request (prompt engineering)\n
        args:
                prompt: Optional[str] = None --- new message (request)
                history: Optional[list[dict[str, str]]] = None --- chatting history [{'role': '<Me>', 'content': 'Hello, how are you?'}, ...] or something like that
                context: Optional[list[dict[str, str] | str]] = None --- could be an example of charater chatting or describing factors
                chat_template: Optional[str] --- jinja template
                add_start: bool = True --- adds a starting point like <character_name> for applying character responce
        \nReturns a string for model input'''
        conversation: list[dict[str, str]] = []
        if context is not None:
            if isinstance(context, str):
                conversation.append({'role': self.system_name, 'content': context})
            else:
                conversation += context
        if history is not None:
            conversation += history
        if prompt is not None:
            if isinstance(prompt, str):
                conversation.append({'role': self.user_name, 'content': prompt})
        if len(conversation) == 0:
            return ''
        if chat_template is None:
            chat_template = self.chat_template #type: ignore
        if chat_template is None:
            ans: str = ''
            for message in conversation:
                ans += message['role'] + ' ' + message['content'] + ' '
        else:
            ans = self.llm.tokenizer.apply_chat_template(conversation=conversation, chat_template=chat_template, tokenize=False) #type: ignore
        if add_start:
            ans += self.my_name + ' '
        return ans


class StoppingCriteria:
    '''Stoping Criteria - list of of words that stops text generation'''

    def __init__(self, device: torch.device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 stop_words: Iterable[str | np.ndarray | torch.Tensor]):
        '''Creates Stopping Criteria parameter\n
        args:
                device: torch.device --- device which contains list of stoping tokens
                (must be the same as the model's device)
                tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast --- tokenizer for stoping words
                (must be the same as the model's tokenizer or equivalent one)
                stop_words: Iterable[str | np.ndarray | torch.Tensor] --- list of stop words in string format or tokens format'''
        self.stops: list[torch.Tensor] = []
        self.stops_array: list[np.ndarray] = []
        for word in stop_words:
            if isinstance(word, str):
                word = tokenizer.encode(word, add_special_tokens=False, return_tensors='pt')[0].to(device) # type: ignore
            elif isinstance(word, np.ndarray):
                word = torch.Tensor(word).to(device)
            else:
                if len(word.shape) > 1:
                     word = word.squeeze()
                word = word.to(device)
            for stop in self.stops:
                if torch.all(stop == word):
                    break
            else:
                self.stops.append(word)
                self.stops_array.append(word.cpu().numpy())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        '''Check if there are stop words in generated tokens stream\n
        args:
                input_ids: torch.LongTensor --- tokens stream
                scores: torch.FloatTensor --- not required for now...
        \nReturns True if found one'''
        for stop in self.stops:
            if bool(torch.all((stop == input_ids[0][-stop.shape[0]:]))):
                return True
        return False