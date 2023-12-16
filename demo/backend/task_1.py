from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import spacy
import torch
import os

_directory_path = os.path.dirname(os.path.abspath(__file__))
_models_directory = os.path.join(_directory_path, 'models', '1_predict_domain_and_dialog_act')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class ModelDictionary:
    def __init__(self):
        self.models = {}

    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        else:
            self._load_model(key)
            return self.models[key]

    def _load_model(self, model_id):
        if model_id == 'roberta':
            self.models[model_id] = RoBERTa()
            return
        raise f'Tried to load unknown model with id "{model_id}"'

_models = ModelDictionary()

def predict_domain_and_dialog_act(message, historical_messages, historical_predictions, model_id):
    model = _models[model_id]
    return model.predict(message, historical_messages, historical_predictions)


_labels = ['Hotel-Inform', 'Hotel-Request', 'Restaurant-Inform', 'Restaurant-Request', 'general-bye', 'general-greet', 'general-thank', 'other']
_nlp = spacy.load("en_core_web_lg")
_mlb = MultiLabelBinarizer(classes=_labels)


class RoBERTa(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = len(_labels)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = 256
        self.model_path = os.path.join(_models_directory, 'roberta.pt')
        self.l1 = RobertaForSequenceClassification.from_pretrained(self.model_path, num_labels=self.num_labels)
        self.pre_classifier = torch.nn.Linear(8, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, self.num_labels)
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = output.view(-1, self.num_labels)
        return output
    

    def parse(sentence):
        # Tokenize
        sentence = _nlp(sentence)
        # Remove stop words
        sentence = " ".join([token.lemma_ for token in sentence])
        
        return sentence


    def predict(self, sentence, *_, **__):
        self.eval()
        sentence = self.parse(sentence)
        inputs = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length = self.max_length,
                return_token_type_ids=True,
                padding='max_length',
                return_attention_mask=True,
                truncation=False,
                return_tensors='pt'
            )
        

        input_ids = inputs['input_ids'].to(device, dtype=torch.long)
        attention_mask = inputs['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long)

        outputs = self(input_ids, attention_mask, token_type_ids)

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        threshold = 0.5
        outputs = test_predictions = [[prob > threshold for prob in prob_list] for prob_list in outputs ]
        
        outputs = _mlb.inverse_transform(np.array(outputs))
        return sentence, outputs