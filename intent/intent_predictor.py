from LSTM import LSTM
import os

LSTM_MODEL_PATH = 'saved_models'
LSTM_MODEL_NAME = 'LSTM_BERT_HISTORY'

class IntentPredictor:
    def __init__(self, model_path, model_name):
        # TODO: Load model and tokenizer
        self.intent_classifier = LSTM(model_path, model_name, cuda = True)
        self.preprocessor = self.intent_classifier.preprocessor

    def predict(self, utterance):
        preprocessed_utterance = self.preprocessor.preprocess(utterance)
        intent = self.intent_classifier.predict(preprocessed_utterance)
        return intent 