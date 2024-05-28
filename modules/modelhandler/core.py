from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import numpy as np
import os
import pathlib
from typing import NoReturn, Literal, Optional


class LLM:
    '''Large Language Model + Tokenizer\n
    Work explanation "context + prompt" -> encode -> generate -> decode -> "new (generated) text"'''
    
    @staticmethod
    def get_all_gpus() -> dict[int, dict[str, str]] | NoReturn:
        '''Get all available gpus\n
        \nReturns a dictionary with index_gpu: parameters'''
        if not torch.cuda.is_available():
            raise Exception('''No gpus available found.
                            Check your cuda drivers (cmd: nvcc --version) and torch with cuda installation.
                            You still can use LLM on your cpu (device = 'cpu'), but this is not recomended.''')
        gpus = {}
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            gpus[i] = {'name': device.name, 'total_memory': device.total_memory,
                       'multi_processor_count': device.multi_processor_count}
        return gpus
    
    def __init__(self, model_path: str | os.PathLike, device: str | int | torch.device = 'cuda', **kwargs) -> None:
        '''Creates a Large Language Model object (tokenizer included)\n
        args:
                model_path: str | os.PathLike --- path to folder with LLM AWQ model
                device: str | int | torch.device = 'cuda' --- GPU (with cuda) index of gpu (use .get_all_gpus() to see all avaliable)
        kwargs: --- default settings for generating
                do_sample: bool = True
                stopping_criteria: Optional[StoppingCriteriaList | list[StoppingCriteria]] = None
                temperature: float = 0.8
                top_p: float = 0.2
                top_k: int = 40
                max_new_tokens: int = 64'''
        self._model_path = pathlib.Path(model_path)
        self.model_name = self._model_path.name
        self._device = self._get_device(device)
        self.model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True, 
                                                       trust_remote_code=False, safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self._default_generate_params = {'do_sample': kwargs.get('do_sample', True),
                                         'stopping_criteria': kwargs.get('stopping_criteria'),
                                         'temperature': kwargs.get('temperature', 0.8),
                                         'top_p': kwargs.get('top_p', 0.2),
                                         'top_k': kwargs.get('top_k', 40),
                                         'max_new_tokens': kwargs.get('max_new_tokens', 64)}

    def _get_device(self, device: Optional[str | int | torch.device] = None) -> torch.device:
        '''Method that casts all kind of device designations to torch.device\n
        args:
                device: Optional[str | int | torch.device] = None --- if None returns default
        \nReturns torch.device'''
        if device:
            return torch.device(device) if isinstance(device, str) or isinstance(device, int) else device
        return self._device

    def encode(self, prompt: str, add_special_tokens: bool = False, 
               tokens_type: Literal['list', 'ndarray', 'Tensor'] = 'list',
               device: Optional[str | int | torch.device] = None) -> list[list[int]] | np.ndarray | torch.Tensor | NoReturn:
        '''Tokenize text (encode to tokens ids)\n
        args:
                prompt: str --- prompt to encode
                add_special_tokens: bool = False --- adding special tokens (such as bos, eos and unk)
                tokens_type: Literal['list', 'ndarray', 'Tensor'] = 'list' --- type of returning tokens
                device: Optional[str | int | torch.device] = None --- which device would handle tokens,
                works only for tokens_type='Tensor'. If is None - load to default device (on a same device as model)
        \nReturns tokenized text in list, np.ndarray or torch.Tensor'''
        match tokens_type:
            case 'list':
                tokens = [self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)]
            case 'Tensor':
                tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens, return_tensors='np')
                tokens = tokens.astype(np.int32) #type: ignore
            case 'ndarray':
                tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens, return_tensors='pt')
                tokens = torch.type(torch.int32).to(self._get_device(device)) #type: ignore
        return tokens
    
    def decode(self, tokens: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
               skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True) -> str:
        '''Detokenize text (decode to string format)\n
        args:
                tokens: list[int] | np.ndarray | torch.Tensor --- tokens that need to be decoded
                skip_special_tokens: bool = False --- skipping special tokens (such as bos, eos and unk)
                clean_up_tokenization_spaces: bool = True --- delete extra spaces (if possible)
        \nReturns a decoded string (text)'''
        if isinstance(tokens, list):
            if len(tokens) == 0:
                return ''
            if isinstance(tokens[0], list):
                tokens = tokens[0]
        else:
            if len(tokens.shape) == 2:
                tokens = tokens.squeeze(0)
        output = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens,
                                       clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        return output
    
    def generate(self, tokens: list[int] | list[list[int]] | np.ndarray | torch.Tensor, **kwargs) -> torch.Tensor:
        '''Generate new text in tokenized format\n
        args:
                tokens: torch.Tensor --- tokenized text
        kwargs: --- default settings for generating
                do_sample: bool = True
                stopping_criteria: Optional[StoppingCriteriaList | list[StoppingCriteria]] = None
                temperature: float = 0.8
                top_p: float = 0.2
                top_k: int = 40
                max_new_tokens: int = 64
        \nReturns tokens'''
        if isinstance(tokens, np.ndarray) or isinstance(tokens, list):
            tokens = torch.Tensor(tokens).type(torch.int32).to(self._device)
        elif tokens.device != self._device:
            tokens = tokens.to(self._device)
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        params = self._default_generate_params.copy()
        params.update(kwargs)
        output = self.model.generate(tokens,
                                     do_sample = params['do_sample'],
                                     stopping_criteria = params['stopping_criteria'],
                                     temperature = params['temperature'],
                                     top_p = params['top_p'],
                                     top_k = params['top_k'],
                                     max_new_tokens = params['max_new_tokens'])[0]
        return output
