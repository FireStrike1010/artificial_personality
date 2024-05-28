import asyncio
from os import walk
from pathlib import Path
import logging
import re
import numpy as np
from typing import NoReturn

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, CallbackQueryHandler, filters

from modules.interface import ChatMemory, Character
from modules.modelhandler import LLM, ChatBot

from .user import User

class TelegramBot:
    def __init__(self, token: str) -> None:
        self.__token = token
        self._allowed_usernames: set[str] | None = None
        self.users: dict[str, User] = dict()
        self.characters: dict[str, Character] = dict()
        self._characters_path: set[Path] = set()
    
    def set_saving(self, saves_folder: Path | str, autosave_time_min: int | None = None) -> None:
        if isinstance(saves_folder, str):
            saves_folder = Path(saves_folder)
        self.saves_folder = saves_folder
    
    async def save(self) -> None:
        import pickle
        from datetime import datetime
        file_name = Path(datetime.now().strftime(format='%Y%m%d_%H%M%S')+'.pickle') #type: ignore
        with open(self.saves_folder/file_name, mode='wb') as file:
            pickle.dump(self.users, file)
    
    def set_allowed_usernames(self, allowed_usernames: set[str]):
        self._allowed_usernames = allowed_usernames

    async def add_allowed_username(self, allowed_username: str):
        allowed_username = allowed_username.replace('@', '', 1)
        if self._allowed_usernames is None:
            self._allowed_usernames = set([allowed_username])
        else:
            self._allowed_usernames.add(allowed_username)

    def build(self) -> None:
        self.application = Application.builder().token(self.__token).build()
        self.application.add_handler(CommandHandler('start', self.start))
        self.application.add_handler(CallbackQueryHandler(self.callback))
        self.application.add_handler(CommandHandler('help', self.help))
        self.application.add_handler(CommandHandler('settings', self.settings))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.generate))
        self.logger.info("Application was built successfully")

    def set_logger(self, print_logging: bool = True, logging_messages: bool = False, logs_folder: str | None = None) -> None:
        if print_logging:
            self.logger = logging.getLogger(__name__)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            streamhandler = logging.StreamHandler()
            streamhandler.setFormatter(formatter)
            self.logger.addHandler(streamhandler)
            level = logging.DEBUG if logging_messages else logging.INFO
            self.logger.setLevel(level)
        else:
            self.logger = logging
            self.logger.disable()
    
    def set_LLM(self, chatbot: str | ChatBot, device: str = 'cuda', default_settings: dict[str, int | float] = {}) -> None:
        self.default_settings = {'temperature': default_settings.get('temperature', 0.8),
                                 'top_p': default_settings.get('top_p', 0.8),
                                 'top_k': default_settings.get('top_k', 40),
                                 'max_new_tokens': default_settings.get('max_new_tokens', 64),
                                 'memory_size': default_settings.get('memory_size', 10)}
        self.logger.info(f"Loading LLM into {device}...")
        if isinstance(chatbot, ChatBot):
            self.chatbot = chatbot
        else:
            model = LLM(chatbot, device=device)
            self.chatbot = ChatBot(model)
        self.logger.info(f"LLM ({self.chatbot.llm.model_name}) loaded successfully")

    def add_character(self, path: str | Path, tokenize: bool = True) -> None:
        if tokenize and not hasattr(self, 'chatbot'):
            raise ValueError('The TelegramBot does not have chatbot, use set_LLM() to add one')
        path = Path(path) if isinstance(path, str) else path
        if path.is_dir():
            characters_paths = []
            for dirpath, _, filenames in walk(path):
                for file in filenames:
                    characters_paths.append(Path(dirpath, file))
        else:
            characters_paths = [path]
        characters_paths = set(filter(lambda x: x.suffix in ('.json', '.yaml', '.png'), characters_paths))
        characters_paths = characters_paths - self._characters_path
        for path in characters_paths:
            try:
                character = Character(path)
                self.logger.info(f"Character card ({str(path)}) loaded successfully")
                self.characters[character.card['name']] = character
                self._characters_path.add(path)
            except:
                self.logger.error(f"Character card ({str(path)}) didn't load")
    
    def run(self) -> None | NoReturn:
        if not hasattr(self, 'application'):
            raise AttributeError('Telegram Bot (application) was not built use .build()')
        if not hasattr(self, 'chatbot'):
            raise AttributeError('Telegram Bot do not have an LLMHandler use .set_LLM()')
        self.logger.info("Running a Telegram Bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def _check_availability(self, update: Update) -> bool:
        if update.effective_user is None or update.message is None:
            self.logger.error(f'''Missing user or message''')
            return False
        if self._allowed_usernames is None or update.effective_user.username in self._allowed_usernames:
            return True
        await update.message.reply_text('System: Access is denied...')
        self.logger.warning(f"Access is denied for @{update.effective_user.username}")
        return False

    async def _get_user(self, update: Update) -> User | None:
        if update.effective_user is None or update.effective_user.username is None:
            self.logger.error(f'''Missing user''')
            return
        user = self.users.get(update.effective_user.username)
        if user is None:
            self.logger.info(f"New user @{update.effective_user.username} detected")
            return
        return user

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_availability(update):
            return
        user = await self._get_user(update)
        if update.effective_user is None or update.effective_user.username is None or update.message is None:
            self.logger.error(f'''Missing user or message''')
            return
        if user is None:
            user = User(update.effective_user.username, update.effective_user.first_name, self.default_settings)
            self.users[update.effective_user.username] = user
            self.logger.info(f"New user @{update.effective_user.username} registered")
        commands = [['/start'], ['/help'], ['/settings']]
        commands = ReplyKeyboardMarkup(commands)
        temp_message = await update.message.reply_text('Loading...', reply_markup=commands)
        character_names = self.characters.keys()
        buttons = [[InlineKeyboardButton(i, callback_data='select_character ' + i)] for i in character_names]
        buttons = InlineKeyboardMarkup(buttons)
        user.current_callback = await update.message.reply_text('Select character:', reply_markup=buttons)
        await context.bot.delete_message(temp_message.chat_id, temp_message.message_id)

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._get_user(update)
        if user is None or user.current_callback is None or update.effective_user is None:
            self.logger.error(f'''Missing user in database''')
            return
        if update.callback_query is None or update.callback_query.data is None:
            self.logger.error(f'''Missing callback query''')
            return
        if context._chat_id is None:
            self.logger.error(f'''Missing chat id in context''')
            return
        data = update.callback_query.data
        if re.search(r'select_character\s', data):
            data = re.sub(r'select_character\s', '', data)
            await self.select_character(update, context, data)
            await context.bot.delete_message(user.current_callback.chat_id, user.current_callback.message_id)
            user.current_callback = None
        elif data == 'settings_back':
            await context.bot.delete_message(user.current_callback.chat_id, user.current_callback.message_id)
            user.current_callback = None
            user.current_callback_message = None
            user.current_callback_message_type = None
        elif data == 'temperature':
            user.current_callback_message_type = data
            user.current_callback_message = await context.bot.send_message(context._chat_id, f'''
System:
Temperature: Controls randomness, higher values increase diversity. (Higher temperature will make outputs more random and diverse.)
Current: {user.generate_settings['temperature']}                                                                                                                                                        
Type new floating point value between 0.0 and ∞ (standart - between 0.0 and 2.0)''')
        elif data == 'top_p':
            user.current_callback_message_type = data
            user.current_callback_message = await context.bot.send_message(context._chat_id, f'''
System:
Top-p (nucleus): The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus. (Lower top-p values reduce diversity and focus on more probable tokens.)
Current: {user.generate_settings['top_p']}
Type new floating point value between 0.0 and 1.0''')
        elif data == 'top_k':
            user.current_callback_message_type = data
            user.current_callback_message = await context.bot.send_message(context._chat_id, f'''
System:
Top-k: Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens. (Lower top-k also concentrates sampling on the highest probability tokens for each step.)
Current: {user.generate_settings['top_k']}
Type new integer value between 0 and ∞ (standart - between 10 and 100)''')
        elif data == 'max_new_tokens':
            user.current_callback_message_type = data
            user.current_callback_message = await context.bot.send_message(context._chat_id, f'''
System:
Generate tokens: Number of maximum tokens to generate. (Higher values means more words generated.)
Current: {user.generate_settings['max_new_tokens']}
Type new integer value between 0 and ∞ (standart - between 50 and 1000)''')
        elif data == 'memory_size':
            user.current_callback_message_type = data
            user.current_callback_message = await context.bot.send_message(context._chat_id, f'''
System:
Memory size: Number of messages used for generating. (Higher values means more messages that character remebers)
Current: {user.generate_settings['memory_size']}
Type new integer value between 0 and ∞ (standart - between 5 and 10, if you want to all - type "all")''')
        else:
            self.logger.error(f"Recived unmatched callback ({data}) from @{update.effective_user.username}")
        
    async def select_character(self, update: Update, context: ContextTypes.DEFAULT_TYPE, data) -> None:
        user = await self._get_user(update)
        if user is None:
            self.logger.error(f'''Missing user''')
            return
        if update.callback_query is None:
            self.logger.error(f'''Missing calback query''')
            return
        character_name = self.characters[data].card['name']
        if user.current_character != character_name:
            memory = user.memories.get(character_name)
            if memory is None:
                memory = ChatMemory()
                user.memories[character_name] = memory
            user.current_memory = memory
            user.current_character = character_name
        await update.callback_query.answer(f'Selected {character_name}')
        self.logger.info(f"User @{user.username} selected {character_name}")
        await self.send_first_message(update, context)

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._get_user(update)
        if update.message is None:
            self.logger.error(f'''Missing message''')
            return
        if user is None:
            await update.message.reply_text('System: Use /start to select a character')
            return
        settings = user.generate_settings
        buttons = [[InlineKeyboardButton(f"Temperature: {settings['temperature']}", callback_data='temperature')], # type: ignore
                   [InlineKeyboardButton(f"Top-p: {settings['top_p']}", callback_data='top_p')], # type: ignore
                   [InlineKeyboardButton(f"Top-k: {settings['top_k']}", callback_data='top_k')], # type: ignore
                   [InlineKeyboardButton(f"Generate tokens: {settings['max_new_tokens']}", callback_data='max_new_tokens')], # type: ignore
                   [InlineKeyboardButton(f"Memory size: {settings['memory_size']}", callback_data='memory_size')], # type: ignore
                   [InlineKeyboardButton('« Back', callback_data='settings_back')]]
        buttons = InlineKeyboardMarkup(buttons)
        user.current_callback = await update.message.reply_text('Select setting that you want to change:', reply_markup=buttons)
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            self.logger.error(f'''Missing message''')
            return
        await update.message.reply_text('''
System:
/help - this command
/start - select character from avaliable
/settings - change settings (temperature, top-p, top-k, generate tokens)''')

    async def generate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._get_user(update)
        if update.message is None or update.message.text is None:
            self.logger.error(f'''Missing message''')
            return
        if user is None or user.current_character is None or user.current_memory is None:
            await update.message.reply_text('System: Use /start to select a character')
            return
        if user.current_callback_message:
            value = update.message.text
            match user.current_callback_message_type:
                case 'temperature':
                    value = abs(float(value))
                    user.generate_settings['temperature'] = value # type: ignore
                case 'top_p':
                    value = abs(float(value))
                    user.generate_settings['top_p'] = value # type: ignore
                case 'top_k':
                    value = abs(int(value))
                    user.generate_settings['top_k'] = value # type: ignore
                case 'max_new_tokens':
                    value = abs(int(value))
                    user.generate_settings['max_new_tokens'] = value # type: ignore
                case 'memory_size':
                    try:
                        value = abs(int(value))
                    except:
                        value = 'all'
                    user.generate_settings['memory_size'] = value # type: ignore
            self.logger.info(f"User @{user.username} changed the {user.current_callback_message_type} setting on {value}")
            user.current_callback_message_type = None
            try:
                await context.bot.delete_message(user.current_callback_message.chat_id, user.current_callback_message.message_id)
                await context.bot.delete_message(user.current_callback.chat_id, user.current_callback.message_id) # type: ignore
            except:
                pass
            await self.settings(update, context)
        else:
            text = update.message.text
            character = self.characters.get(user.current_character)
            self.logger.debug(f"""@{user.username} to {character.card['name']}: "{text}" """) # type: ignore
            responce = self.chatbot(prompt=text, history=user.current_memory.get_history(self.default_settings.get('memory_size', 'all')), # type: ignore
                                    context=character.card['context'], **user.generate_settings) # type: ignore
            user.current_memory.add_message(responce['input']['role'], responce['input']['content'])
            user.current_memory.add_message(responce['output']['role'], responce['output']['content'])
            await update.message.reply_text(responce['output']['content'])
            self.logger.debug(f"""{character.card['name']} to @{user.username}: "{responce['output']['content']}" """) #type: ignore

    async def send_first_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._get_user(update)
        character = self.characters.get(user.current_character) #type: ignore
        greeting = character.card['greeting'] #type: ignore
        user.current_memory.add_message(self.chatbot.my_name, greeting) #type: ignore
        await context.bot.send_message(context._chat_id, greeting) #type: ignore
        self.logger.debug(f"""{character.card['name']} to @{user.username}: "{greeting}" """) #type: ignore
