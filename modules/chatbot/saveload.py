from zipfile import ZipFile
import pickle
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from modules.chatbot.chatbot import ChatBot
from modules.chatbot.chatmemory import ChatMemory
from modules.chatbot.character import Character


def save(chatbot: ChatBot) -> None:
    load_dotenv()
    cache_folder = os.getenv("cahce_folder")
    saves_folder = os.getenv("saves")
    name = str(datetime.now().date())
    character_cache_file = Path(f'{cache_folder}/character_{name}.bin')
    memory_cache_file = Path(f'{cache_folder}/memory_{name}.bin')
    with open(memory_cache_file, 'wb') as temp_file:
        pickle.dump(chatbot.memory, temp_file)
    with open(character_cache_file, 'wb') as temp_file:
        pickle.dump(chatbot.character)
    archive_path = Path(saves_folder + f'/{name}.zip')
    with ZipFile(archive_path, 'w') as archive:
        archive.write(memory_cache_file)
        archive.write(character_cache_file)
    os.remove(memory_cache_file)
    os.remove(character_cache_file)

def load(date: str = None) -> tuple[ChatMemory, Character]:
    memory = None
    character = None
    load_dotenv()
    cache_folder = os.getenv("cahce_folder")
    saves_folder = os.getenv("saves")
    archives = os.listdir(saves_folder)
    print(archives)
    if archives == []:
        return None, None
    if date is None:
        archives = sorted(archives)[::-1]
        for i in archives:
            if '.zip' in archives:
                archive = i
                break
    else:
        archive = Path(f'{saves_folder}/{date}.zip')
        if not os.access(archive):
            raise FileExistsError(f'File {date}.zip doesnt exists')
    with ZipFile(archive) as archive:
        archive.extractall(cache_folder)
    for file in os.listdir(cache_folder):
        if 'character_' in file:
            path = Path(f'{cache_folder}/{file}')
            character = pickle.load(path)
            os.remove(path)
        elif 'memory_' in file:
            path = Path(f'{cache_folder}/{file}')
            memory = pickle.load(path)
            os.read(path)
    return memory, character
        
        