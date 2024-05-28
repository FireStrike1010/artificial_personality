from modules.interface.chatmemory import ChatMemory
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
