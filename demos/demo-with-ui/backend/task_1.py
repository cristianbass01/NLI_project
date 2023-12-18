from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import os
from dataset_and_task_exploration.util import ModelDictionary, models_base_directory, device, spacy_nlp

_models_directory = os.path.join(models_base_directory, '1_predict_domain_and_dialog_act')

_models = ModelDictionary({'roberta' : lambda: RoBERTaPredictDialogAct()})

def predict_domain_and_dialog_act(message, historical_messages, historical_domain_and_dialog_acts, model_id):
    model = _models[model_id]
    parsed_sentence, prediction = model.predict(message, historical_messages, historical_domain_and_dialog_acts)
    return parsed_sentence, prediction



class RoBERTaPredictDialogAct(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_file = torch.load(os.path.join(_models_directory, 'roberta', 'model.pt'))
        self.mlb = MultiLabelBinarizer(classes=model_file['mlb']['classes'], sparse_output=model_file['mlb'].get('sparse_output', False))
        self.mlb.fit([model_file['mlb']['classes']])
        self.num_labels = len(self.mlb.classes)
        self.l1 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = 256
        self.pre_classifier = torch.nn.Linear(8, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, self.num_labels)
        self.load_state_dict(model_file['state_dict'])
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
    

    def parse(self, sentence):
        # Tokenize
        sentence = spacy_nlp(sentence)
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

        outputs = outputs.cpu().detach().numpy()
        threshold = 0
        outputs = [[prob > threshold for prob in prob_list] for prob_list in outputs ]
        
        outputs = self.mlb.inverse_transform(np.array(outputs))
        return sentence, outputs