import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from sklearn.preprocessing import MultiLabelBinarizer
from torch import cuda
import re
import fake_database as FakeDatabase
#import nltk
#nltk.download('punkt')
from transformers import logging
logging.set_verbosity_error()

import spacy
nlp = spacy.load("en_core_web_lg")

SAVED_MODELS_DIR = "./saved_models/"
DIALOGUE_ACT_MODEL_PATH = SAVED_MODELS_DIR + "1_model_dialog_act.pt"
SLOT_FILLING_MODEL_PATH = SAVED_MODELS_DIR + "2_model_slot_filling.pt"
QUESTION_TAGS_MODEL_PATH = SAVED_MODELS_DIR + "2_2_model_question_tags_no_none.pt"
TO_BE_RETRIEVED_MODEL_PATH = SAVED_MODELS_DIR + "3_1_model_to_be_retrieved.pt"
AGENT_DIALOGUE_ACT_MODEL_PATH = SAVED_MODELS_DIR + "3_2_model_agent_dialog_act.pt"
TO_BE_REQUESTED_MODEL_PATH = SAVED_MODELS_DIR + "3_3_model_to_be_requested.pt"



class Agent:
    def __init__(self):
        # Initialize any necessary variables or resources here
        self.history = []
        self.index = -1

        self.dialogue_act_model = None
        self.dialogue_act_mlb = None
        self.slot_filling_model = None
        self.slot_filling_index2tag = None
        self.question_tags_model = None
        self.question_tags_mlb = None
        self.to_be_retrieved_model = None
        self.to_be_retrieved_mlb = None
        self.agent_dialogue_act_model = None
        self.agent_dialogue_act_mlb = None
        self.to_be_requested_model = None
        self.to_be_requested_mlb = None

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer_slots = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

        
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        
        self.database = FakeDatabase.FakeDatabase()
        self.load_modules()
        pass

    def reset_history(self):
        self.historical_utt = []
        return

    def add_user_utterance(self, utterance):
        # Update the history with the user's utterance
        self.index += 1
        self.history.append({'user_utterance': parse(utterance)})
        return

    def add_agent_utterance(self, utterance):
        # Update the history with the agent's utterance
        self.history[self.index]['agent_utterance'] = parse(utterance)
        return


    def predict_dialogue_act(self):
        # Predict the dialogue act of the utterance
        historical_utterance = self.get_dialogue_acts_historical_utterance()
        sentence, prediction = self.predict_multi_label(self.dialogue_act_model, self.dialogue_act_mlb, historical_utterance)
        self.history[self.index]['dialogue_act'] = [item for tup in prediction for item in tup]
        return self.history[self.index]['dialogue_act']
    
    def predict_slot_filling(self):
        # Predict the slot filling of the utterance
        historical_utterance = self.get_slot_filling_historical_utterance()
        sentence, prediction = self.predict_slots(historical_utterance)
        
        self.history[self.index]['slot_filling'] = {}
        for index in range(len(sentence)):
            if prediction[index] != 'O':
                slot_name = prediction[index][2:]
                slot_value = map_slot_value(sentence[index])
                # decide what to mantain and what to remove (same day, same people, etc.)
                if self.index > 0 and slot_name in self.history[self.index-1]['slot_filling']:
                    if slot_value == 'same':
                        slot_value = self.history[self.index-1]['slot_filling'][slot_name]
                    
                if prediction[index].startswith('I'):
                    if slot_name in self.history[self.index]['slot_filling']:
                        slot_value = self.history[self.index]['slot_filling'][slot_name] + ' ' + slot_value
                
                self.history[self.index]['slot_filling'][slot_name] = map_slot_value(slot_value)
        
        return self.history[self.index]['slot_filling']
    
    def predict_question_tags(self):
        # Predict the question tags of the utterance
        user_dialogue_act = self.history[self.index]['dialogue_act']
        user_dialogue_act = process_dialogue_act(user_dialogue_act)
        user_dialogue_act = [act_type for act_type in user_dialogue_act if act_type.endswith('Request')]
        if len(user_dialogue_act) == 0:
            self.history[self.index]['question_tags'] = []
            return self.history[self.index]['question_tags']
        
        historical_utterance = self.get_question_tag_historical_utterance()
        sentence, prediction = self.predict_multi_label(self.question_tags_model, self.question_tags_mlb, historical_utterance)
        self.history[self.index]['question_tags'] = prediction
        return prediction
    
    def predict_to_be_retrieved(self):
        #current_act_type = process_dialogue_act(self.history[self.index]['dialogue_act'])
        #current_user_booking = [act_type.split('-')[0].lower() for act_type in current_act_type if not act_type.startswith('other')]
        #if len(current_user_booking) == 0:
        #    self.history[self.index]['to_be_retrieved'] = []
        #    return self.history[self.index]['to_be_retrieved']
        
        # Predict the to be retrieved of the utterance
        historical_utterance = self.get_to_be_retrieved_historical_utterance()
        sentence, prediction = self.predict_multi_label(self.to_be_retrieved_model, self.to_be_retrieved_mlb, historical_utterance)
        self.history[self.index]['to_be_retrieved'] = [item for tup in prediction for item in tup]
        if prediction != ['none']:
            self.history[self.index]['to_be_provided'] = self.get_fake_retrieved(self.history[self.index]['to_be_retrieved'])
        return self.history[self.index]['to_be_retrieved']

    def predict_agent_dialogue_act(self):
        # Predict the agent dialogue act of the utterance
        historical_utterance = self.get_agent_dialogue_acts_historical_utterance()
        sentence, prediction = self.predict_multi_label(self.agent_dialogue_act_model, self.agent_dialogue_act_mlb, historical_utterance)
        self.history[self.index]['agent_dialogue_act'] = [item for tup in prediction for item in tup]
        return self.history[self.index]['agent_dialogue_act']

    def predict_to_be_requested(self):
        current_act_type = process_dialogue_act(self.history[self.index]['agent_dialogue_act'])
        requests = [act_type for act_type in current_act_type if act_type.endswith('Request')]
                
        if len(requests) == 0:
            self.history[self.index]['to_be_requested'] = []
            return self.history[self.index]['to_be_requested']
        
        # Predict the to be requested of the utterance
        historical_utterance = self.get_to_be_requested_historical_utterance()
        sentence, prediction = self.predict_multi_label(self.to_be_requested_model, self.to_be_requested_mlb, historical_utterance)
        self.history[self.index]['to_be_requested'] = [item for tup in prediction for item in tup]
        return self.history[self.index]['to_be_requested']

    def get_agent_response(self):
        to_be_retrieved = self.history[self.index]['to_be_retrieved']
        to_be_provided = self.history[self.index]['to_be_provided']
        agent_dialogue_act = self.history[self.index]['agent_dialogue_act']
        response = self.database.retrieve_agent_response(agent_dialogue_act, to_be_retrieved, to_be_provided)
        self.history[self.index]['agent_utterance'] = parse(response)
        return response

    def get_fake_retrieved(self, to_be_retrieved):
        retrieved = {}
        for slot in to_be_retrieved:
            if 'availability' not in slot:
                retrieved[slot] = self.database.retrieve(slot)
        
        domains = set()
        for slot in to_be_retrieved:
            domains.add(slot.split('-')[0])
            
        for domain in domains:
            if len(to_be_retrieved) != 0 and any((slot_name_value.split(":")[0]!=domain+"-none" for slot_name_value in list(retrieved) + [''])):
                retrieved[domain + '-availability'] = 'yes'
            else:
                retrieved[domain + '-availability'] = 'no'
        
        return retrieved

    def get_dialogue_acts_historical_utterance(self):
        if self.index == 0:
            prev_user_utterance = ''
            prev_user_acts = []
            prev_bot_utterance = ''
            prev_bot_acts = []
        else:
            prev_user_utterance = self.history[self.index-1]['user_utterance']
            prev_user_acts = self.history[self.index-1]['dialogue_act']
            prev_bot_utterance = self.history[self.index-1]['agent_utterance']
            prev_bot_acts = self.history[self.index-1]['agent_dialogue_act']
        
        current_user_utterance = self.history[self.index]['user_utterance']

        historical_utterance = ' | '.join([prev_user_utterance, ', '.join(prev_user_acts), prev_bot_utterance, ', '.join(prev_bot_acts), current_user_utterance])
        return historical_utterance
    
    def get_slot_filling_historical_utterance(self):
        if self.index == 0:
            prev_user_acts = []
            prev_bot_acts = []
        else:
            prev_user_acts = self.history[self.index-1]['dialogue_act']
            prev_bot_acts = self.history[self.index-1]['agent_dialogue_act']
        
        current_user_acts = self.history[self.index]['dialogue_act']

        prev_user_acts_str = " , ".join(prev_user_acts)
        prev_bot_acts_str = " , ".join(prev_bot_acts)
        current_user_acts_str = " , ".join(current_user_acts)
        current_user_utterance = self.history[self.index]['user_utterance']
        
        historical_utterance = ' | '.join([prev_user_acts_str, prev_bot_acts_str, current_user_acts_str, current_user_utterance])
        historical_utterance = historical_utterance.split()

        return historical_utterance

    def get_question_tag_historical_utterance(self):
        if self.index == 0:
            prev_user_utterance = ''
            prev_user_acts = []
            prev_slots = []
            prev_bot_utterance = ''
            prev_bot_acts = []
            prev_bot_slots = []
        else:
            prev_user_utterance = self.history[self.index-1]['user_utterance']
            prev_user_acts = self.history[self.index - 1]['dialogue_act']
            prev_slots = self.history[self.index - 1]['slot_filling'].keys() 
            prev_slots += [item + ' : ?' for item in self.history[self.index - 1]['question_tags']]
            prev_bot_utterance = self.history[self.index - 1]['agent_utterance']
            prev_bot_acts = self.history[self.index - 1]['agent_dialogue_act']
            prev_bot_slots = self.retrieve_slots_from_agent_act(prev_bot_acts)
            
        current_user_utterance = self.history[self.index]['user_utterance']
        current_user_acts = self.history[self.index]['dialogue_act']

        prev_user_acts_str = " | ".join(prev_user_acts)
        prev_slots_str = " | ".join(prev_slots)
        prev_bot_acts_str = " | ".join(prev_bot_acts)
        prev_bot_slots_str = " | ".join(prev_bot_slots)
        current_user_acts_str = " | ".join(current_user_acts)
        
        historical_utterance = ' | '.join([prev_user_utterance, prev_user_acts_str, prev_slots_str, 
                                        prev_bot_utterance, prev_bot_acts_str, prev_bot_slots_str, 
                                        current_user_utterance, current_user_acts_str])
        return historical_utterance

    def retrieve_slots_from_agent_act(self, prev_bot_acts):
        prev_bot_acts_processed = process_dialogue_act(prev_bot_acts).remove('other')
        services = list(set([item.split('-')[0] for item in prev_bot_acts_processed]))
        prev_bot_raw_slots = self.history[self.index - 1]['to_be_retrieved']
        prev_bot_slots = [item for item in prev_bot_raw_slots if not (item.endswith('availability') or item.endswith('choice'))]
        if len(services) == 0: 
            prev_bot_slots = [services[0] + item[len('booking'):] if item.startswith('booking') else item for item in prev_bot_slots]
        return prev_bot_slots
    
    def get_to_be_retrieved_historical_utterance(self):
        if self.index == 0:
            prev_user_utterance = ''
            prev_user_acts = []
            prev_user_act_type_to_slots = {}
        else:
            prev_user_utterance = self.history[self.index-1]['user_utterance']
            prev_user_acts = self.history[self.index - 1]['dialogue_act']
            prev_user_slots = self.history[self.index - 1]['slot_filling']
            for item in self.history[self.index - 1]['question_tags']:
                prev_user_slots[item] = '?'
            prev_user_act_type_to_slots = self.get_slots_per_act_type(prev_user_acts, prev_user_slots)

        user_utterance = self.history[self.index]['user_utterance']
        user_acts = self.history[self.index]['dialogue_act']
        user_slots = self.history[self.index]['slot_filling']
        for item in self.history[self.index]['question_tags']:
            user_slots[item] = '?'
            
        user_act_type_to_slots = self.get_slots_per_act_type(user_acts, user_slots)

        user_booking_service = list(set([item.split('-')[0].lower() for item in process_dialogue_act(user_acts)]))
        
        historical_utterance = concatenate_user_act_type_slots(prev_user_utterance, prev_user_act_type_to_slots)
        historical_utterance += concatenate_user_act_type_slots(user_utterance, user_act_type_to_slots)
        historical_utterance += " | " + " , ".join(user_booking_service)

        return historical_utterance

    def get_slots_per_act_type(self, acts, slots):
        slots_per_act_type = {}
        for slot in slots.keys():
            matching_act_types = [act for act in acts if slot.split('-')[0] in act.lower()]
            for act in acts:
                domain = slot.split('-')[0]
                if domain in act.lower():
                    if (slots[slot] == '?' and 'request' in act.lower()) or (slots[slot] != '?' and 'inform' in act.lower()):
                        matching_act_types.append(act)
                
            if len(matching_act_types) == 0:
                matching_act_types = ['Restaurant-Inform'] if 'restaurant' in slot.split('-')[0] else ['Hotel-Inform']
            matching_act_type = matching_act_types[0]
            
            if matching_act_type not in slots_per_act_type:
                slots_per_act_type[matching_act_type] = [(slot.split('-')[1], slots[slot])]
            else:
                slots_per_act_type[matching_act_type].append((slot.split('-')[1], slots[slot]))
        
        return slots_per_act_type
    
    def get_agent_dialogue_acts_historical_utterance(self):
        if self.index == 0:
            prev_agent_utterance = ''
            prev_agent_acts_to_slot = {}
        else:
            prev_agent_utterance = self.history[self.index - 1]['agent_utterance']
            prev_agent_acts = self.history[self.index - 1]['agent_dialogue_act']
            prev_agent_slots = self.history[self.index - 1]['to_be_provided']
            prev_agent_slots.update(self.history[self.index - 1]['to_be_requested'])
            prev_agent_acts_to_slot = self.get_slots_per_act_type(prev_agent_acts, prev_agent_slots)

        prev_user_acts = self.history[self.index]['dialogue_act']
        prev_user_slots = self.history[self.index]['slot_filling']
        for item in self.history[self.index - 1]['question_tags']:
            prev_user_slots[item] = '?'
        prev_user_utterance = self.history[self.index]['user_utterance']

        prev_user_acts_to_slot = self.get_slots_per_act_type(prev_user_acts, prev_user_slots)
        prev_user_booking_service = process_dialogue_act(prev_user_acts)

        agent_to_be_retrieved = self.history[self.index]['to_be_retrieved']

        historical_utterance = concatenate_user_act_type_slots(prev_user_utterance, prev_user_acts_to_slot)
        historical_utterance += " | " + concatenate_user_act_type_slots(prev_agent_utterance, prev_agent_acts_to_slot)
        historical_utterance += " | " + " , ".join(prev_user_booking_service)
        historical_utterance += " | " + " , ".join(agent_to_be_retrieved)
        return historical_utterance
    
    def get_to_be_requested_historical_utterance(self):
        if self.index == 0:
            prev_user_utterance = ''
            prev_user_acts = []
            prev_user_act_type_to_slots = {}
        else:
            prev_user_utterance = self.history[self.index-1]['user_utterance']
            prev_user_acts = self.history[self.index - 1]['dialogue_act']
            prev_user_slots = self.history[self.index - 1]['slot_filling']
            for item in self.history[self.index - 1]['question_tags']:
                prev_user_slots[item] = '?'
            prev_user_act_type_to_slots = self.get_slots_per_act_type(prev_user_acts, prev_user_slots)

        user_utterance = self.history[self.index]['user_utterance']
        user_acts = self.history[self.index]['dialogue_act']
        user_slots = self.history[self.index]['slot_filling']
        for item in self.history[self.index]['question_tags']:
            user_slots[item] = '?'
            
        user_act_type_to_slots = self.get_slots_per_act_type(user_acts, user_slots)

        user_booking_service = list(set([item.split('-')[0].lower() for item in process_dialogue_act(user_acts)]))
        
        to_be_provided = self.history[self.index]['to_be_provided']

        historical_utterance = concatenate_user_act_type_slots(prev_user_utterance, prev_user_act_type_to_slots)
        historical_utterance += concatenate_user_act_type_slots(user_utterance, user_act_type_to_slots)
        historical_utterance += " | " + " , ".join(user_booking_service)
        historical_utterance += " | "
        historical_utterance += " , ".join(to_be_provided)
        return historical_utterance



    def get_dialogue_act_of(self, index):
        return self.history[index]['dialogue_act']
    
    def get_slot_filling_of(self, index):
        return self.history[index]['slot_filling']
    
    def get_question_tags_of(self, index):
        return self.history[index]['question_tags']
    
    def get_to_be_retrieved_of(self, index):
        return self.history[index]['to_be_retrieved']
    
    def get_agent_dialogue_act_of(self, index):
        return self.history[index]['agent_dialogue_act']
    
    def get_to_be_requested_of(self, index):
        return self.history[index]['to_be_requested']



    def load_modules(self):
        # Load the multilabel modules here
        self.dialogue_act_model, self.dialogue_act_mlb = self.load_multi_label_model(DIALOGUE_ACT_MODEL_PATH)
        self.question_tags_model, self.question_tags_mlb = self.load_multi_label_model(QUESTION_TAGS_MODEL_PATH)
        self.to_be_retrieved_model, self.to_be_retrieved_mlb = self.load_multi_label_model(TO_BE_RETRIEVED_MODEL_PATH)
        self.agent_dialogue_act_model, self.agent_dialogue_act_mlb = self.load_multi_label_model(AGENT_DIALOGUE_ACT_MODEL_PATH)
        self.to_be_requested_model, self.to_be_requested_mlb = self.load_multi_label_model(TO_BE_REQUESTED_MODEL_PATH)
        
        # Load the slot filling module here
        self.slot_filling_model, self.slot_filling_index2tag = self.load_slot_filling(SLOT_FILLING_MODEL_PATH)
        return
    
    def load_multi_label_model(self, checkpoint_fpath):
        print('Loading model: {}'.format(checkpoint_fpath))
        checkpoint = torch.load(checkpoint_fpath, map_location=self.device)
        mlb = MultiLabelBinarizer(**checkpoint['mlb'])
        mlb.fit(checkpoint['mlb']['classes'])
        model = BERTClass(len(mlb.classes))
        model = model.to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model, mlb
        
    def load_slot_filling(self, checkpoint_fpath):
        print('Loading model: {}'.format(checkpoint_fpath))
        checkpoint = torch.load(checkpoint_fpath, map_location=self.device)
        index2tag = checkpoint['mlb']
        model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(index2tag))
        model.to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model, index2tag

    def predict_slots(self, sentence):
        self.slot_filling_model.eval()
        inputs = self.tokenizer_slots(sentence,
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors="pt")
        # move to gpu
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        # forward pass
        outputs = self.slot_filling_model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.slot_filling_model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = self.tokenizer_slots.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.slot_filling_index2tag[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
        return sentence, prediction
    
    def predict_multi_label(self, model, mlb, sentence):
        model.eval()
        sentence = parse(sentence)
        inputs = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length = 256,
                return_token_type_ids=True,
                padding='max_length',
                return_attention_mask=True,
                truncation=False,
                return_tensors='pt'
            )
        

        input_ids = inputs['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
        token_type_ids = inputs['token_type_ids'].to(self.device, dtype=torch.long)

        outputs = model(input_ids, attention_mask, token_type_ids)

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        threshold = 0.5
        outputs = [[prob > threshold for prob in prob_list] for prob_list in outputs ]
        
        outputs = mlb.inverse_transform(np.array(outputs))
        return sentence, outputs

def parse(sentence):
    # Tokenize
    sentence = nlp(sentence)
    # Remove stop words
    sentence = " ".join([token.lemma_ for token in sentence])
    
    return sentence

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClass, self).__init__()
        self.num_labels = num_labels
        self.l1 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels)
        self.pre_classifier = torch.nn.Linear(self.num_labels, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, self.num_labels)

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
    
def process_dialogue_act(service_list):
    services = set()
    if len(service_list) == 0:
        services.add('other')
    for service in service_list:
        if service.startswith('Restaurant') or service.startswith('restaurant'):
            services.add(service)
        elif service.startswith('Hotel') or service.startswith('hotel'):
            services.add(service)
        elif service.startswith('booking') or service.startswith('Booking'):
            services.add(service)
        elif service.startswith('general'):
            services.add(service)
        else:
            services.add('other')
    return sorted(list(services))

def concatenate_user_act_type_slots(user_utterance, user_act_type_to_slots):

    historical_utterance = user_utterance + " | "
    for act_type in user_act_type_to_slots:
        historical_utterance += act_type + " = "
        for slot_name, slot_value in user_act_type_to_slots[act_type]:
            historical_utterance += slot_name + " : " + slot_value + " , "
        
        if len(user_act_type_to_slots[act_type]) > 0:
            # Remove last comma
            historical_utterance = historical_utterance[:-3]

        historical_utterance += " ; "

    if len(user_act_type_to_slots) > 0:
        # Remove last semicolon
        historical_utterance = historical_utterance[:-3]

    return historical_utterance


def map_slot_value(input: str):
    input = input.lower()

    number_map = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    
    output = input
    if input in number_map:
        output = number_map[input]
    
    if input == 'center':
        return 'centre'
    
    if 'any' in input:
        return 'dontcare'
    
    regex = re.compile('not[ \t]*matter')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('doesn\'?t[A-Za-z \t]*matter')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('not?[A-Za-z \t]*preference')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('not?[ \t]*particular(ly)?')
    if regex.search(input):
        return 'dontcare'

    regex = re.compile('don\'?t[A-Za-z \t]*preference')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('don\'?t[A-Za-z \t]*care')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('not[ \t]*care')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('not[ \t]*really')
    if regex.search(input):
        return 'dontcare'
    
    regex = re.compile('[A-Za-z \t]*same[A-Za-z \t]*')
    if regex.search(input):
        return 'same'
    
    return output


print('Loading agent...')
agent = Agent()
print('Agent loaded!')

sentence = 'i need a place to dine in the center thats expensive'
print('Adding user utterance:', sentence)
agent.add_user_utterance(sentence)
print('Historical utterance:', agent.get_dialogue_acts_historical_utterance())
print('Predicting dialogue act...')
print('Predicted dialogue act:', agent.predict_dialogue_act())
print('History:', agent.history)
print()
print('Predicting slot filling...')
print('Historical slot filling utterance:', agent.get_slot_filling_historical_utterance())
print('Predicted slot filling:', agent.predict_slot_filling())
print('History:', agent.history)
print()
print('Predicting question tags...')
print('Historical question tags utterance:', agent.get_question_tag_historical_utterance())
print('Predicted question tags:', agent.predict_question_tags())
print('History:', agent.history)
print()
print('Predicting to be retrieved...')
print('Historical to be retrieved utterance:', agent.get_to_be_retrieved_historical_utterance())
print('Predicted to be retrieved:', agent.predict_to_be_retrieved())
print('History:', agent.history)
print()
print('Predicting agent dialogue act...')
print('Historical agent dialogue act utterance:', agent.get_agent_dialogue_acts_historical_utterance())
print('Predicted agent dialogue act:', agent.predict_agent_dialogue_act())
print('History:', agent.history)
print()
print('Predicting to be requested...')
print('Historical to be requested utterance:', agent.get_to_be_requested_historical_utterance())
print('Predicted to be requested:', agent.predict_to_be_requested())
print('History:', agent.history)
print()
