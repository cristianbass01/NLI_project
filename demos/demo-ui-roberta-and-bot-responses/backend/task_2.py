from transformers import RobertaForTokenClassification, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import spacy
import torch
import os
from collections import defaultdict
from util.generate_metrics_latex_table import ModelDictionary, models_base_directory, device, spacy_nlp

_models_directory = os.path.join(models_base_directory, '2_slot_filling')

_models = ModelDictionary({'roberta' : lambda: RoBERTaSemanticFrameSlotFilling()})

def predict_semantic_frame_slot_filling(message, historical_messages, historical_domain_and_dialog_acts, historical_slot_values, historical_slot_questions, model_id):
    model = _models[model_id]
    return model.predict(message, historical_messages, historical_domain_and_dialog_acts, historical_slot_values, historical_slot_questions)


class RoBERTaSemanticFrameSlotFilling():
    def __init__(self):
        self.slot_filling_model = RoBERTaSlotFilling()
        self.question_labeling_model = RoBERTaQuestionLabeling()

    def predict(self, message, historical_messages, historical_domain_and_dialog_acts, historical_slot_values, historical_slot_questions):
        # Call the slot filling model
        words, bio_tags = self.slot_filling_model.predict(message)

        slot_values = self.convert_bio_tags(words, bio_tags)
        
        sentence, slot_questions = self.question_labeling_model.predict(message)

        return slot_values, slot_questions
    
    def convert_bio_tags(self, words, bio_tags):
        slot_dict = defaultdict(list)

        for word, bio_tag in zip(words, bio_tags):
            if bio_tag.startswith('B-') or bio_tag.startswith('I-'):
                key = bio_tag[2:]
                slot_dict[key].append(word)
        
        return {key: " ".join(value) for key, value in slot_dict.items()}

    

class RoBERTaSlotFilling():
    def __init__(self):
        # Load model for slot filling
        model_file = torch.load(os.path.join(_models_directory, 'roberta', '2_1_model_bio_tagging.pt'), map_location=device)
        self.index2tag = model_file['mlb']
        labels = [self.index2tag[k] for k in sorted(self.index2tag)]
        self.mlb = MultiLabelBinarizer(classes=labels)
        self.mlb.fit([labels])

        self.model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(self.mlb.classes))
        self.model.load_state_dict(model_file['state_dict'])
        self.model.to(device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

        self.max_length = 128
    
    def parse(self, sentence):
        # Tokenize
        sentence = spacy_nlp(sentence)
        # Remove stop words
        sentence = " ".join([token.lemma_ for token in sentence])
        
        return sentence

    def predict(self, sentence, *_, **__):
        self.model.eval()
        words = self.parse(sentence).split()
        inputs = self.tokenizer(words,
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt")
        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        # forward pass
        outputs = self.model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.index2tag[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
        return words, prediction
    



class RoBERTaQuestionLabeling(torch.nn.Module):
    def __init__(self):
        super(RoBERTaQuestionLabeling, self).__init__()
        model_file = torch.load(os.path.join(_models_directory, 'roberta', '2_2_model_question_labeling.pt'))
        self.mlb = MultiLabelBinarizer(classes=model_file['mlb']['classes'], sparse_output=model_file['mlb'].get('sparse_output', False))
        self.mlb.fit([model_file['mlb']['classes']])
        self.num_labels = len(self.mlb.classes)
        self.l1 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels)
        self.max_length = 256
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.pre_classifier = torch.nn.Linear(self.num_labels, 768)
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
        output = output.view(-1, self.num_labels)  # Reshape the output
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
        outputs = [output for output_tuple in outputs for output in output_tuple]
        return sentence, outputs