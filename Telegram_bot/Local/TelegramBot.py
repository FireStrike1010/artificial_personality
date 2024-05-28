import os
from dotenv import load_dotenv
load_dotenv()
from modules.telegram.bothandler import TelegramBot


api_token = os.getenv('api_token')
if api_token is None:
    raise ValueError('Specify the api_key in .env file')

character_folder_path = os.getenv('character_folder_path')

allowed_usernames_path = os.getenv('allowed_usernames_path')
if allowed_usernames_path is not None:
    if os.path.exists(allowed_usernames_path):
        allowed_usernames = []
        with open(allowed_usernames_path) as file:
            for line in file:
                allowed_usernames.append(line.replace('@', '', 1))
        allowed_usernames = set(allowed_usernames)
        if len(allowed_usernames) == 0:
            allowed_usernames = None
    else:
        allowed_usernames = None
    
saves_folder = os.getenv('saves_folder')
autosave_time_min = os.getenv('autosave_time_min')
if autosave_time_min is not None:
    autosave_time_min = int(autosave_time_min)

print_logging = bool(os.getenv('logging'))
logging_messages = bool(os.getenv('logging_messages'))
logs_folder_path = os.getenv('logs_folder_path', './logs')

model_path_or_link = os.getenv('model_path_or_link')
if model_path_or_link is None:
    raise ValueError('Specify the model_path_or_link in .env file')
device = os.getenv('device')
if device is None:
    device = 'cuda:0'

default_settings = {}
default_temperature = os.getenv('temperature')
if default_temperature is not None:
    default_settings['temperature'] = float(default_temperature)
default_top_p = os.getenv('top_p')
if default_top_p is not None:
    default_settings['top_p'] = float(default_top_p)
default_top_k = os.getenv('top_k')
if default_top_k is not None:
    default_settings['top_k'] = int(default_top_k)
default_max_new_tokens = os.getenv('max_new_tokens')
if default_max_new_tokens is not None:
    default_settings['max_new_tokens'] = abs(int(default_max_new_tokens))
default_memory_size = os.getenv('defualt_memory_size')
if default_memory_size is not None:
    default_settings['memory_size'] = abs(int(default_memory_size))


def main() -> None:
    tb = TelegramBot(api_token)
    
    tb.set_logger(print_logging=print_logging, logging_messages=logging_messages, logs_folder=logs_folder_path)
    if allowed_usernames:
        tb.set_allowed_usernames(allowed_usernames)
    tb.build()
    tb.set_LLM(model_path_or_link, device, default_settings)
    if character_folder_path:
        tb.add_character(character_folder_path)
    if saves_folder:
        tb.set_saving(saves_folder, autosave_time_min)
    tb.run()

if __name__ == '__main__':
    main()