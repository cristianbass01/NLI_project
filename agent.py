import sys
sys.path.append('intent')
sys.path.append('slots')

import slots.slot_filler as SF
import intent.intent_predictor as INT

INT_MODEL_NAME = 'LSTM_BERT_HISTORY'
INT_MODEL_PATH = 'intent/saved_models'

SF_MODEL_NAME = 'roberta-base'
SF_MODEL_PATH = 'slots/saved_models'

# INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
# print(INT_model.predict('I want to book a hotel and also want a table for 2 at a restaurant in the city center for 2 people'))

# SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, cuda = True)
# print(SF_model.tag_slots('Do not care how many people'))

class Agent():
    def __init__(self, intent_use_history = False, slot_use_history = False):
        self.INT_model = INT.IntentPredictorLSTM(INT_MODEL_PATH, INT_MODEL_NAME, cuda = False)
        # TODO: keep only slots that match intent (I think so, idk)
        self.SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, cuda = True)
        self.utterance_history = []
        self.acts_history = []
        self.intent_use_history = intent_use_history
        self.slot_use_history = slot_use_history
    
    def predict_intent(self, utterance, given_utterance_history = None, given_acts_history = None):
        utterance_history = given_utterance_history if given_utterance_history is not None else self.utterance_history
        acts_history = given_acts_history if given_acts_history is not None else self.acts_history
        
        if self.intent_use_history:
            composed_prefix = ' | '.join([utterance_history[-2], ', '.join(acts_history[-2]), utterance_history[-1], ', '.join(acts_history[-1])]) + ' | '
            utterance = composed_prefix + utterance
        return self.INT_model.predict(utterance)
    
    def predict_slots(self, utterance):
        self.predict_intent(utterance)
        self.SF_model.tag_slots(utterance)
    
    def respond(self, input):
        pass