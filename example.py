from modules.chatbot.chatbot import *
import textwrap
import sys

args = sys.argv[1:]
if len(args) == 0:
    raise Exception('Type a path to model folder')
if len(args) == 2:
    ch = Character(args[1])
    cb = ChatBot(args[0], character=ch)
else:
    cb = ChatBot(args[0])

name, res = cb.start()
print(textwrap.fill(f'{name}: {res}', width = 80))

while True:
    inp = input('User: ')
    if inp == '':
        break
    name, res = cb.send_message(inp)
    name, res = cb.get_responce()
    print(textwrap.fill(f'{name}: {res}', width=80))