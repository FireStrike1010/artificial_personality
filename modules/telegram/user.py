from modules.telegram.chatmemory import ChatMemory
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from telegram import Message
from typing import Optional

class User:
    def __init__(self, username: Optional[str] = None, first_name: Optional[str] = None, generate_settings: Optional[dict] = None) -> None:
        self.username = username
        self.first_name = first_name
        self.generate_settings = generate_settings
        self.current_callback: Message | None = None
        self.current_callback_message: Message | None = None
        self.current_callback_message_type: str | None = None
        self.current_memory: ChatMemory | None = None
        self.current_character: str | None = None
        self.memories: dict[str, ChatMemory] = {}
    
    def _retokenize(self, tokenizer_name: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        for memory in self.memories.values():
            if memory._tokenizer_name != tokenizer_name:
                memory._retokenize(tokenizer_name, tokenizer)