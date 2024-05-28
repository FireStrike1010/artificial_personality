# Artificial Personality
Artificial Personality is text2text AI chatbot that can use character cards <br>
Supports only awq models for now <br>
Recommended for less than 6 gb vram - mistral-7b-awq (https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-AWQ) <br>
Recommended for less than 12gb vram - llama-2-13b-awq (https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ) <br>
Works on Windows and Linux-based OSes <br>
Base requirements: NVIDIA Cuda 12.1 and Python 3.10 <br>
## Instalation:
1. Create virtual enviroment - python env "any_name"
3. Copy all files from this repository (using "git clone" or manualy) to that enviroment folder
2. Activate virtual enviroment - ./Scripts/Activate.ps1 (Scripts/activate for linux)
4. Install libraries - pip install -r ./requirements.txt
## Running a server
1. Activate virtual enviroment - ./Scripts/Activate.ps1 (Scripts/activate for linux)
2. Launch a Fast Api Server with Uvicorn - python ./server/run_server.py
3. Go to ip:port/docs to see query schema
4. Use ip:port/query for generating
## Running a Telegram Bot
1. Activate virtual enviroment - ./Scripts/Activate.ps1 (Scripts/activate for linux)
2. Launch a bot - python ./Telegram_Bot/Local/run_server.py
3. Type a /start to @YourBotUsername to select a character or /help to see all commands
