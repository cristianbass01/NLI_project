import os
import pickle
from LSTM import MyLSTM as LSTM
from dict_model import DictModel

LSTM_MODEL_PATH = 'saved_models'
LSTM_MODEL_NAME = 'LSTM_BERT_HISTORY'

class ToBeRetrievedPredictor:
    def __init__(self, model_path, model_name):
        self.input_mlb = pickle.load(open(os.path.join(model_path, 'MOVE_RETR_input_mlb.pkl'), 'rb'))
        self.output_mlb = pickle.load(open(os.path.join(model_path, 'MOVE_RETR_output_mlb.pkl'), 'rb'))
        self.predictor = pickle.load(open(os.path.join(model_path, 'MOVE_RETR_' + model_name + '.pkl'), 'rb'))

    def predict(self, slots_per_act_type):
        model_inputs = []
        to_be_retrieved = []
        for act_type in slots_per_act_type:
            domain = act_type.split("-")[0].lower()
            slots = slots_per_act_type[act_type]
            
            if len(slots) != 0:
                to_be_retrieved.append(domain + '-availability')
            
            for slot in slots:
                model_input = act_type.lower() + '-' + slot[0]
                model_inputs.append(model_input)
        
        model_output = self.predictor.predict(self.input_mlb.transform([model_inputs]))
        to_be_retrieved = to_be_retrieved + list(self.output_mlb.inverse_transform(model_output)[0])
        return to_be_retrieved