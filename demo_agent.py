from colorama import Fore, Back
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dialogue_file')
parser.add_argument('--debug', action = 'store_true')

args = parser.parse_args()

dialogue_file = args.dialogue_file
debug = args.debug

user_inputs = [utterance[len('USR: ') :] for utterance in open(dialogue_file, 'r').read().split('\n') if utterance.startswith('USR: ')]
bot_responses = [utterance[len('BOT: ') :] for utterance in open(dialogue_file, 'r').read().split('\n') if utterance.startswith('BOT: ')]
cur_response = 0
cur_input = 0

print("Importing agent...")
from agent import Agent
print("Loading agent...")
agent = Agent(intent_use_history = True, slot_use_history = True)
print("Agent loaded!\n")

while True:
    if not debug:
        user_input = input("User's input: ")
    else:
        user_input = user_inputs[cur_input]
        print("User's input: ", user_input)
        cur_input += 1
    
    intent = agent.predict_intent(user_input)
    print(Back.RED + "User intents:" + Back.RESET + " " + Fore.RED + ", ".join(intent) + Fore.RESET)
    
    print()
    print(Back.RED + "User filled slots:" + Back.RESET)
    slots = agent.predict_slots(user_input)
    for slot in slots:
        print('\t' + Fore.BLUE + slot[0] + ': ' + Fore.RED + slot[1] + Fore.RESET)
    print()
    
    print(Back.RED + "Agent Move Prediction:" + Back.RESET)
    
    slots_per_act_type = agent.get_slots_per_act_type(intent, slots)
    to_be_retrieved = agent.predict_to_be_retrieved(slots_per_act_type)
    print("\t" + Back.RED + "To be retrieved:" + Back.RESET + " " + Fore.RED + ", ".join(to_be_retrieved) + Fore.RESET)
    
    retrieved = agent.get_fake_retrieved(to_be_retrieved)
    
    agent_acts = agent.predict_agent_acts(user_input, slots_per_act_type, retrieved)
    print("\t" + Back.RED + "Agent intents:" + Back.RESET + " " + Fore.RED + ", ".join(agent_acts) + Fore.RESET)
    
    to_be_req = agent.predict_to_be_requested(user_input, slots_per_act_type, retrieved)
    print("\t" + Back.RED + "To be requested:" + Back.RESET + " " + Fore.RED + ", ".join(to_be_req) + Fore.RESET)
    print('\n')
    
    agent.update_history(intent, user_input)
    
    #TODO: Get scripted bot response according to some set dialogue structure (e.g. a list of responses)
    bot_response = bot_responses[cur_response]
    cur_response += 1
    print("Agent's response:", bot_response)
    agent.update_history(agent_acts, bot_response)
    
    if cur_response == len(bot_responses):
        break
    

print("Exiting...")