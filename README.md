# Artificial Personality
Artificial Personality is text2text AI chatbot that can use character cards <br>
Supports only awq models for now <br>
Recommended for less than 6 gb vram - mistral-7b-awq (https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-AWQ) <br>
Recommended for less than 12gb vram - llama-2-13b-awq (https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ) <br>
Works on Windows and Linux-based OSes <br>
Requirements for use: NVIDIA Cuda 12.1 and Python 3.10 <br>
## Instalation:
1. Create virtual enviroment - python env "any_name"
3. Copy all files from this repository (using "git clone" or manualy) to that enviroment folder
2. Activate virtual enviroment - Scripts/Activate.ps1 (Scripts/activate for linux)
4. Install libraries - pip install requirements.txt
5. Run example - python example.py "path_to_model_folder" "path_to_character_card"
