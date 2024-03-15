from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import numpy as np
from pathlib import Path

class LLMHandler:
    '''Large Language Model + Tokenizer\n
    Work explanation "text + context" -> encode -> generate -> decode -> "new (generated) text"\n
    init: 
         model_path: str | pathlib.Path --- path to folder with LLM AWQ model
         device: str | torch.device = 'cuda' --- GPU (with cuda) index to load model (use .get_all_gpus() to see all avaliable)
         kwargs: --- default settings for generating
                temperature: float = 0.8
                top_p: float = 0.2
                top_k: int = 40
                max_new_tokens: int = 32 --- max generated output (length in tokens (words))
    del: --- deleting model and tokenizer from gpu memory and ram'''
    
    @staticmethod
    def get_all_gpus() -> dict:
        gpus = {}
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            gpus[i] = {'name': device.name, 'total_memory': device.total_memory,
                       'multi_processor_count': device.multi_processor_count}
        return gpus
    
    def __init__(self, model_path: str | Path, device: str | torch.device = 'cuda', **kwargs) -> None:
        self._model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.model_name = self._model_path.name
        self._device = torch.device(device) if isinstance(device, str) else device
        self.model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True,
                                                         trust_remote_code=False, safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)        
        self._default_generate_params = {'temperature': kwargs.get('temperature', 0.8),
                                         'top_p': kwargs.get('top_p', 0.2),
                                         'top_k': kwargs.get('top_k', 40),
                                         'max_new_tokens': kwargs.get('max_new_tokens', 64)}
        
    def encode(self, prompt: str, add_special_tokens: bool = True) -> np.ndarray:
        '''encode --- tokenize text (encode to tokens ids)\n
        prompt: str --- prompt to encode
        add_special_tokens: bool = True --- adding special tokens (such as "<s>", "</s>" and "<unk>")
        \nreturns tokenized text in np.ndarray'''
        prompt = np.array(self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)).astype(np.int32)
        return prompt
    
    def decode(self, tokens: list[int] | np.ndarray | torch.Tensor,
               skip_special_tokens: bool = False) -> str:
        '''decode --- detokenize text (decode to string format)\n
        tokens: list[int] | np.ndarray | torch.Tensor --- tokens that need to be decoded
        skip_special_tokens: bool = False --- skipping special tokens (such as "<s>", "</s>" and "<unk>")
        \nreturns a decoded string (text)'''
        skip_special_tokens = bool(skip_special_tokens)
        output = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return output
    
    def generate(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        '''generate --- generate new text in tokenized format\n
        tokens: torch.Tensor --- tokenized text
        kwargs: --- default settings for generating
               temperature: float = 0.8
               top_p: float = 0.2
               top_k: int = 40
               max_new_tokens: int = 32 --- max generated output (length in tokens (words))'''
        tokens = tokens.unsqueeze(0)
        params = self._default_generate_params.copy()
        params.update(kwargs)
        output = self.model.generate(tokens, do_sample=True,
                                     temperature = params['temperature'],
                                     top_p = params['top_p'],
                                     top_k = params['top_k'],
                                     max_new_tokens = params['max_new_tokens'])[0].cpu()
        return output
