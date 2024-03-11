from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from pathlib import Path
import yaml
import json

class Character:
    '''A character initializated from tavern card\n
    init:
        path_to_tavern_card: str | Path = None --- path to tavern card file (default - create a basic assistant)
        tokenizer_name: str = None --- name of model tokenizer'''
    
    def __init__(self, path_to_tavern_card: str | Path = None, tokenizer_name: str = None) -> None:
        self._tokenizer_name = tokenizer_name
        self.card = {}
        if path_to_tavern_card is None:
            self._create_base_assistant()
            return
        if isinstance(path_to_tavern_card, str):
            path_to_tavern_card = Path(path_to_tavern_card)
        match path_to_tavern_card.suffix:
            case '.json':
                with open(path_to_tavern_card, 'r') as file:
                    card = json.load(file)
            case '.yaml':
                with open(path_to_tavern_card, 'r') as file:
                    card = yaml.safe_load(file)
            case '.png':
                raise Exception('Dont work right now, mb in next versions...')
            case _:
                raise Exception('Unsupported file type for tavern card, supports only .json, .yaml and .png files')
        card = card.get('data', card)
        self.card['name'] = card.get('name', 'Assistant')
        self.card['description'] = card.get('creator_notes', '')
        self.card['greeting'] = card.get('first_mes', 'Hey! I am your personal assistant. I will help you with anything.')
        context = card.get('description', '')
        context = context.replace('{{user}}', 'You')
        context = context.replace('{{char}}', 'Me')
        self.card['context'] = context
    
    def _create_base_assistant(self) -> None:
        '''_create_base_assistant --- creates a basic assistant instead of character'''
        self.card = {}
        self._tokenized_context = None
        self.card['name'] = 'Assistant'
        self.card['description'] = 'Assistant is very useful character for any work.'
        self.card['greeting'] = 'Hey! I am your personal assistant. I will help you with anything.'
        self.card['context'] = ''
    
    def _retokenize(self, tokenizer_name: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                    add_special_tokens: bool = True) -> None:
        '''_retokenize --- tokenize of retokenize first context
        tokenizer_name: str --- new name of model tokenizer
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast --- new tokenizer
        add_special_tokens: bool = True --- adding special tokens (such as "<s>", "</s>" and "<unk>")'''
        self._tokenizer_name = tokenizer_name
        if (self.card['context'] is None) or (self.card['context'] == ''):
            self._tokenized_context = np.array([]).astype(np.int32)
        else:
            self._tokenized_context = np.array(tokenizer.encode(self.card['context'], add_special_tokens=add_special_tokens)).astype(np.int32)