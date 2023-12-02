import sys
sys.path.append('intent')
sys.path.append('slots')

import intent.intent_predictor as INT
import slots.slot_filler as SF
import slots.retrieve_slots_predictor as RETR

INT_MODEL_NAME = 'LSTM_BERT_HISTORY'
INT_MODEL_PATH = 'intent/saved_models'

SF_MODEL_NAME = 'roberta-base'
SF_MODEL_SUFFIX = 'with_intent'
SF_MODEL_PATH = 'slots/saved_models'

RETR_MODEL_NAME = 'LSTM_BERT_HISTORY'
RETR_MODEL_PATH = 'slots/saved_models'

# INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
# print(INT_model.predict('I want to book a hotel and also want a table for 2 at a restaurant in the city center for 2 people'))

# SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, SF_MODEL_SUFFIX, cuda = True)
# print(SF_model.tag_slots('Do not care how many people'))
# print(SF_model.tag_slots("I want to book a hotel and also want a table for 2 at a restaurant in the city center for 2 people"))

class Agent():
    def __init__(self, intent_use_history = False, slot_use_history = False):
        self.INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
        self.SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, SF_MODEL_SUFFIX, cuda = True)
        self.RETR_model = RETR.RetrieveSlotsPredictorLSTM(RETR_MODEL_PATH, RETR_MODEL_NAME, cuda = False)
        self.utterance_history = []
        self.acts_history = []
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
        # TODO: do we have to use intent to predict slots better?
        # self.predict_intent(utterance)
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
            
    
    def new_dialogue(self):
        self.utterance_history = []
        self.acts_history = []
        self.prev_filled_slots = {}
    
    def respond(self, input):
        pass
