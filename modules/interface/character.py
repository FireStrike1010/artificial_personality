from pathlib import Path
import yaml
import json
from typing import Optional

class Character:
    '''A character initializated from tavern card'''
    
    def __init__(self, path_to_tavern_card: Optional[str | Path] = None) -> None:
        '''Create a Character object\n
        args:
                path_to_tavern_card: Optional[str, Path] = None --- if None - creates a basic assistant'''
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
        self.file_path = path_to_tavern_card
        card = card.get('data', card)
        self.card['name'] = card.get('name', 'Assistant')
        self.card['description'] = card.get('creator_notes', '')
        self.card['greeting'] = card.get('first_mes', 'Hey! I am your personal assistant. I will help you with anything.')
        context = card.get('description', '')
        context = context.replace('{{user}}', 'You')
        context = context.replace('{{char}}', 'Me')
        self.card['context'] = context
    
    def _create_base_assistant(self) -> None:
        '''Create a basic assistant instead of character'''
        self.card = {}
        self.card['name'] = 'Assistant'
        self.card['description'] = 'Assistant is very useful character for any work.'
        self.card['greeting'] = 'Hey! I am your personal assistant. I will help you with anything.'
        self.card['context'] = ''