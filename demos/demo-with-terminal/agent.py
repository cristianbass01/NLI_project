import sys
sys.path.append('1_intent')
sys.path.append('2_slots')
sys.path.append('3_agent_move')

import intent_predictor as INT
import slot_filler as SF
import retrieve_slots_predictor as RETR
import to_be_retrieved_predictor as MOVE_RETR
import agent_acts_predictor as MOVE_AGENT_ACTS
import to_be_requested_predictor as MOVE_AGENT_REQ

from fake_database import FakeDatabase

INT_MODEL_NAME = 'LSTM_BERT_HISTORY'
INT_MODEL_PATH = '1_intent/saved_models'

SF_MODEL_NAME = 'roberta-base'
SF_MODEL_SUFFIX = 'with_intent'
SF_MODEL_PATH = '2_slots/saved_models'

RETR_MODEL_NAME = 'LSTM_BERT_HISTORY'
RETR_MODEL_PATH = '2_slots/saved_models'

MOVE_RETR_MODEL_NAME = 'DICT'
MOVE_RETR_MODEL_PATH = '3_agent_move/saved_models'

MOVE_AGENT_ACTS_MODEL_NAME = 'LSTM'
MOVE_AGENT_ACTS_MODEL_PATH = '3_agent_move/saved_models'

MOVE_AGENT_REQ_MODEL_NAME = 'LSTM'
MOVE_AGENT_REQ_MODEL_PATH = '3_agent_move/saved_models'

# INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
# print(INT_model.predict('I want to book a hotel and also want a table for 2 at a restaurant in the city center for 2 people'))

# SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, SF_MODEL_SUFFIX, cuda = True)
# print(SF_model.tag_slots('Do not care how many people'))
# print(SF_model.tag_slots("I want to book a hotel and also want a table for 2 at a restaurant in the city center for 2 people"))

class Agent():
    def __init__(self, intent_use_history = False, slot_use_history = False):
        self.db = FakeDatabase()
        # Intent
        self.INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
        # Slots
        self.SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, SF_MODEL_SUFFIX, cuda = True)
        self.RETR_model = RETR.RetrieveSlotsPredictorLSTM(RETR_MODEL_PATH, RETR_MODEL_NAME, cuda = False)
        # Agent move
        self.MOVE_RETR_model = MOVE_RETR.ToBeRetrievedPredictor(MOVE_RETR_MODEL_PATH, MOVE_RETR_MODEL_NAME)
        self.MOVE_AGENT_ACTS_model = MOVE_AGENT_ACTS.AgentActsPredictor(MOVE_AGENT_ACTS_MODEL_PATH, MOVE_AGENT_ACTS_MODEL_NAME, cuda = False)
        self.MOVE_AGENT_REQ_model = MOVE_AGENT_REQ.ToBeRequestedPredictor(MOVE_AGENT_REQ_MODEL_PATH, MOVE_AGENT_REQ_MODEL_NAME)
        
        self.utterance_history = ['', '']
        self.acts_history = [[], []]
        self.intent_use_history = intent_use_history
        self.slot_use_history = slot_use_history
        self.prev_filled_slots = {}
    
    def get_input_utterance(self, utterance, given_utterance_history, given_acts_history, use_history = False):
        utterance_history = given_utterance_history if given_utterance_history is not None else self.utterance_history
        acts_history = given_acts_history if given_acts_history is not None else self.acts_history
        
        if use_history:
            composed_prefix = ' | '.join([utterance_history[-2], ', '.join(acts_history[-2]), utterance_history[-1], ', '.join(acts_history[-1])]) + ' | '
            utterance = composed_prefix + utterance
        return utterance
        
    
    def predict_intent(self, utterance, given_utterance_history = None, given_acts_history = None):
        utterance = self.get_input_utterance(utterance, given_utterance_history, given_acts_history, self.intent_use_history)
        return self.INT_model.predict(utterance)
    
    def predict_slots(self, utterance, given_utterance_history = None, given_acts_history = None, given_prev_filled_slots = None):
        utterance = self.get_input_utterance(utterance, given_utterance_history, given_acts_history, self.slot_use_history)
        prev_filled_slots = given_prev_filled_slots if given_prev_filled_slots is not None else self.prev_filled_slots

        filled_slots = self.SF_model.tag_slots(utterance)
        # Replace 'same' with previous value
        processed_filled_slots = []
        for slot, slot_value in filled_slots:
            if 'same' in slot_value:
                slot_name = slot.split('-')[-1]
                processed_filled_slots.append((slot, [prev_filled_slots[s] for s in prev_filled_slots if slot_name in s][0]))
            else:
                processed_filled_slots.append((slot, slot_value))
                prev_filled_slots[slot] = slot_value
        # Predict retrieve slots
        retrieve_slots = self.RETR_model.predict(utterance)
        # Remove retrieve slots which have been filled by the slot filler model (trust that model more). Also add question mark
        processed_retrieve_slots = [(slot, '?') for slot in retrieve_slots if slot not in [s[0] for s in processed_filled_slots]]
        
        final_slots = processed_filled_slots + processed_retrieve_slots
        return final_slots
    
    def predict_to_be_retrieved(self, slots_per_act_type):
        return self.MOVE_RETR_model.predict(slots_per_act_type)
    
    def predict_agent_acts(self, user_utterance, user_slots_per_act_type, to_be_retrieved_overall):
        user_slots_per_act_type = [act_type.lower() + '-' + slot[0] + ':' + slot[1]  for act_type in user_slots_per_act_type for slot in user_slots_per_act_type[act_type]]
        to_be_retrieved_overall = [slot + ':' + to_be_retrieved_overall[slot] for slot in to_be_retrieved_overall]
        input_text = user_utterance + ' | USER SLOTS PER ACT ' + ', '.join(user_slots_per_act_type) + ' | RETRIEVED SLOTS ' + ', '.join(to_be_retrieved_overall)
        return self.MOVE_AGENT_ACTS_model.predict(input_text)
    
    def predict_to_be_requested(self, user_utterance, user_slots_per_act_type, to_be_retrieved_overall):
        user_slots_per_act_type = [act_type.lower() + '-' + slot[0] + ':' + slot[1]  for act_type in user_slots_per_act_type for slot in user_slots_per_act_type[act_type]]
        to_be_retrieved_overall = [slot + ':' + to_be_retrieved_overall[slot] for slot in to_be_retrieved_overall]
        input_text = user_utterance + ' | USER SLOTS PER ACT ' + ', '.join(user_slots_per_act_type) + ' | RETRIEVED SLOTS ' + ', '.join(to_be_retrieved_overall)
        return self.MOVE_AGENT_REQ_model.predict(input_text)
    
    def update_history(self, acts, utterance):
        self.utterance_history = [self.utterance_history[-1], utterance]
        self.acts_history = [self.acts_history[-1], acts]
    
    def new_dialogue(self):
        self.utterance_history = ['', '']
        self.acts_history = ['', '']
        self.prev_filled_slots = {}
    
    def get_slots_per_act_type(self, acts, slots):
        slots_per_act_type = {}
        for slot in slots:
            matching_act_types = [act for act in acts if slot[0].split('-')[0] in act.lower()]
            matching_act_types = []
            for act in acts:
                domain = slot[0].split('-')[0]
                if domain in act.lower():
                    if (slot[1] == '?' and 'request' in act.lower()) or (slot[1] != '?' and 'inform' in act.lower()):
                        matching_act_types.append(act)
                
            if len(matching_act_types) == 0:
                matching_act_types = ['Restaurant-Inform'] if 'restaurant' in slot[0].split('-')[0] else ['Hotel-Inform']
            matching_act_type = matching_act_types[0]
            
            if matching_act_type not in slots_per_act_type:
                slots_per_act_type[matching_act_type] = [(slot[0].split('-')[1], slot[1])]
            else:
                slots_per_act_type[matching_act_type].append((slot[0].split('-')[1], slot[1]))
        
        return slots_per_act_type

    def get_fake_retrieved(self, to_be_retrieved):
        retrieved = {}
        for slot in to_be_retrieved:
            if 'availability' not in slot:
                retrieved[slot] = self.db.retrieve(slot)
        
        domains = set()
        for slot in to_be_retrieved:
            domains.add(slot.split('-')[0])
            
        for domain in domains:
            if len(to_be_retrieved) != 0 and any((slot_name_value.split(":")[0]!=domain+"-none" for slot_name_value in list(retrieved) + [''])):
                retrieved[domain + '-availability'] = 'yes'
            else:
                retrieved[domain + '-availability'] = 'no'
        
        return retrieved
    
    def respond(self, input):
        pass

# agent = Agent()
# pred = agent.predict_agent_acts({'Hotel-Inform': [('pricerange', 'expensive')]}, ['hotel-area:all over town', 'hotel-area:center of town', 'hotel-area:except in the north', 'hotel-availability:yes', 'hotel-name:the Gonville'])
# print(pred)