from colorama import Fore, Back

from agent import Agent
print("Loading agent...")
agent = Agent()
print("Agent loaded!\n")

while True:
    user_input = input("User's input: ")
    if 'bye' in user_input.lower():
        break
    agent.add_user_utterance(user_input)
    
    intent = agent.predict_dialogue_act()
    print(Back.RED + "User intents:" + Back.RESET + " " + Fore.RED + ", ".join(intent) + Fore.RESET)
    
    print()
    print(Back.RED + "User filled slots:" + Back.RESET)
    slots = agent.predict_slot_filling()
    questions = agent.predict_question_tags()
    # What I retrieve from slot filling is more important
    questions += slots
    for slot in slots.items():
        print('\t' + Fore.BLUE + slot[0] + ': ' + Fore.RED + slot[1] + Fore.RESET)
    print()
    
    print(Back.RED + "Agent Move Prediction:" + Back.RESET)
    
    to_be_retrieved = agent.predict_to_be_retrieved()
    print("\t" + Back.RED + "To be retrieved:" + Back.RESET + " " + Fore.RED + ", ".join(to_be_retrieved) + Fore.RESET)
    
    agent_acts = agent.predict_agent_dialogue_act()
    print("\t" + Back.RED + "Agent intents:" + Back.RESET + " " + Fore.RED + ", ".join(agent_acts) + Fore.RESET)
    
    to_be_req = agent.predict_to_be_requested()
    print("\t" + Back.RED + "To be requested:" + Back.RESET + " " + Fore.RED + ", ".join(to_be_req) + Fore.RESET)
    print('\n')
    
    bot_response = agent.get_agent_response()
    print("Agent's response:", bot_response)
    

print("Exiting...")