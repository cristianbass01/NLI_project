from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig,
)
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import os
from util import ModelDictionary, models_base_directory, device, spacy_nlp
from agent import Agent
from openai import OpenAI
from collections import defaultdict


agent = Agent()

# Enter Open AI API Key here
openai_api_key = None

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

openai_client = OpenAI(api_key="openai_api_key")

history = defaultdict(list)


def process_user_utterance(message):
    agent.add_user_utterance(message)
    user_dialog_act = agent.predict_dialogue_act()
    slot_values = agent.predict_slot_filling()
    slot_questions = agent.predict_question_tags()
    slots_to_be_retrieved = agent.predict_to_be_retrieved()
    agent_dialog_act = agent.predict_agent_dialogue_act()
    slots_to_be_requested = agent.predict_to_be_requested()

    print(
        f"{user_dialog_act=}, {slot_values=}, {slot_questions=}, {slots_to_be_retrieved=}, {agent_dialog_act=}, {slots_to_be_requested=}"
    )

    history["user_utterance"].append(message)
    history["user_dialog_act"].append(user_dialog_act)
    history["slot_values"].append(slot_values)
    history["slot_questions"].append(slot_questions)
    history["slots_to_be_retrieved"].append(slots_to_be_retrieved)
    history["agent_dialog_act"].append(agent_dialog_act)
    history["slots_to_be_requested"].append(slots_to_be_requested)

    agent_response = generate_response(
        user_dialog_act,
        slot_values,
        slot_questions,
        slots_to_be_retrieved,
        agent_dialog_act,
        slots_to_be_requested,
    )

    agent.add_agent_utterance(agent_response)

    history["agent_response"].append(agent_response)

    result = {
        "user_dialog_act": user_dialog_act,
        "slot_values": slot_values,
        "slot_questions": slot_questions,
        "slots_to_be_retrieved": slots_to_be_retrieved,
        "agent_dialog_act": agent_dialog_act,
        "slots_to_be_requested": slots_to_be_requested,
        "agent_response": agent_response,
    }

    return result


def generate_response(
    user_dialog_act,
    slot_values,
    slot_questions,
    dialog_act,
    slots_to_retrieve,
    slots_to_request,
):
    """
    Generates a response using OpenAI's GPT model, tailored to act like an agent.

    :param slot_values: A dictionary of slot values input by the user.
    :param slot_questions: A list of slot questions asked by the user.
    :param dialog_act: The next agent dialog act.
    :param slots_to_retrieve: A list of slots to be retrieved by the agent.
    :param slots_to_request: A list of slots to be requested of the user by the agent.
    :param api_key: Your OpenAI API key.
    :return: The generated response text.
    """

    history_iterator = zip(history["user_utterance"], history["user_dialog_act"], history["slot_values"], history["slot_questions"], history["slots_to_be_retrieved"], history["agent_dialog_act"], history["slots_to_be_requested"], [*(history["agent_response"]), ''])
    
    historical_data = []
    for user_utterance, user_dialog_act, slot_values, slot_questions, slots_to_retrieve, agent_dialog_act, slots_to_request, agent_response in history_iterator:
        historical_data.append(
f'''
User Utterance: {user_utterance}
User Dialog Act: {user_dialog_act}
Slot Values (user inputs): {slot_values}
Slot Questions (asked by user): {slot_questions}
Slots to Retrieve: {slots_to_retrieve}
Dialog Act you should do: {agent_dialog_act}
Slots to Request:": {slots_to_request}
Your response: {agent_response}
''')
    
    prompt = (
        f"As an AI agent, respond to the user based on the following context:\n"
        f"History:\n{'-------------------'.join(historical_data)}\n----------------\n"
        f"'User Utterance' is the message of the user.\n"
        f"'Slot values (user inputs)' contains relevant values parsed from the user utternace that you should use to respond to the user.\n"
        f"'Slot Questions (asked by user)' are things that the user has requested from you. You have to answer them. Make up information if needed, e.g. for phone numbers.\n"
        f"For 'Slots to Retrieve', creatively generate relevant information (e.g., inventing restaurant names). Do not ask the user for the values of the slots that you should retrieve. However, the values you come up with should align with the slot values that the user has supplied (e.g. suggest relevant hotels/restaurants etc. according to what the user has specified).\n"
        f"For 'Slots to Request', formulate questions to ask the user.\n"
        f"'Dialog Act you should do' is the next dialog act that you should perform.\n"
        f"Furthermore, do not repeat yourself.\n"
        f"Agent Response:"
    )

    # Generating the response

    print(prompt)


    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response.choices[0].message.content
