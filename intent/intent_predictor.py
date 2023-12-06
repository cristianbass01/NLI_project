from LSTM import MyLSTM as LSTM
import os
import pickle
from transformers import AutoTokenizer, BertModel, logging
from torch import load
import torch

logging.set_verbosity_error()

LSTM_MODEL_PATH = 'saved_models'
LSTM_MODEL_NAME = 'LSTM_BERT_HISTORY'
TRANSFORMER_MODEL_NAME = 'bert-base-uncased'

class IntentPredictorLSTM:
    def __init__(self, model_path, model_name, cuda = False):
        self.cuda = cuda
        #Load Tokenizer
        self.nr_features = 768
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
        transformer = BertModel.from_pretrained(TRANSFORMER_MODEL_NAME) 
        self.embedding_matrix = transformer.embeddings.word_embeddings.weight
        # Load MultiLaberBinarizer
        self.mlb = pickle.load(open(os.path.join(model_path, 'mlb.pkl'), 'rb'))
        self.intent_classifier = LSTM(input_size = self.nr_features, num_cells = 4, hidden_size = 300, bi = True, out_features = len(self.mlb.classes_))
        self.intent_classifier.load_state_dict(load(os.path.join(model_path, 'INT_' + model_name + '.pt')))
        if cuda:
            self.intent_classifier = self.intent_classifier.cuda()
        self.intent_classifier.eval()

    def predict(self, utterance):
        with torch.no_grad():
            tokenized = self.tokenizer(utterance)
            embedding = self.embedding_matrix[tokenized.input_ids]
            if self.cuda:
                embedding = embedding.cuda()
            out = self.intent_classifier(embedding[None, :])
            if self.cuda:
                out = out.cpu()
            out = (out > 0).detach().numpy()
            intents = self.mlb.inverse_transform(out)[0]
            return intents