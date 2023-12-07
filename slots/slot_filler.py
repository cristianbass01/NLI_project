import pandas as pd
from transformers import AutoTokenizer, RobertaForTokenClassification
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import re

TRANSFORMER_MODEL_NAME = 'roberta-base'
TRANSFORMER_MODEL_SUFFIX = 'with_intent'
MODEL_PATH = 'saved_models'

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
    
    return output
    
class SlotFiller():
    def __init__(self, model_path, model_name, model_suffix, cuda = False):
        self.cuda = cuda
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # !!! Make sure to change this as well if tags are changed
        self.possible_tags = ['B-hotel-area', 'B-hotel-bookday', 'B-hotel-bookpeople', 'B-hotel-bookstay', 'B-hotel-name', 'B-hotel-pricerange', 'B-hotel-stars', 'B-hotel-type', 'B-restaurant-area', 'B-restaurant-bookday', 'B-restaurant-bookpeople', 'B-restaurant-booktime', 'B-restaurant-food', 'B-restaurant-name', 'B-restaurant-pricerange', 'I-hotel-area', 'I-hotel-bookday', 'I-hotel-bookpeople', 'I-hotel-name', 'I-hotel-pricerange', 'I-hotel-type', 'I-restaurant-area', 'I-restaurant-bookday', 'I-restaurant-bookpeople', 'I-restaurant-booktime', 'I-restaurant-food', 'I-restaurant-name', 'I-restaurant-pricerange', 'O']
        self.transformer = RobertaForTokenClassification.from_pretrained(os.path.join(model_path, 'SF_' + model_name + '_' + model_suffix), num_labels = len(self.possible_tags))
        if self.cuda:
            self.transformer = self.transformer.cuda()
        self.transformer.eval()
    
    def tag_slots(self, utterance):
        tokenizer = self.tokenizer
        transformer = self.transformer
        possible_tags = self.possible_tags
        
        tokenized = tokenizer(utterance, return_tensors = 'pt', padding = 'max_length')
        if self.cuda:
            tokenized.input_ids = tokenized.input_ids.cuda()
            tokenized.attention_mask = tokenized.attention_mask.cuda()
        
        out = transformer.forward(input_ids = tokenized.input_ids, attention_mask = tokenized.attention_mask)
        pred = torch.argmax(out.logits, dim = 2).squeeze()
        if self.cuda:
            pred = pred.cpu()
        # tag name, value
        tags = {}
        in_tag = False
        last_pred_word = -1
        for i in range(pred.shape[0]):
            span_has_ended = False
            if tokenized.token_to_word(i) is not None and tokenized.token_to_word(i) != last_pred_word:
                tag = possible_tags[pred[i]]
                
                if tag != 'O':
                    if not in_tag:
                        in_tag = True
                        cur_type = tag[2 :]
                        span_start = tokenized.token_to_chars(i)[0]
                    elif cur_type != tag[2 :]:
                        span_has_ended = True
                elif in_tag:
                    span_has_ended = True
            
            elif tokenized.token_to_word(i) is None and in_tag:
                span_has_ended = True
            
            if span_has_ended:
                in_tag = False
                span_end  = tokenized.token_to_chars(i - 1)[1]
                # Trust the first tag found more
                if cur_type not in tags:
                    tags[cur_type] = utterance[span_start : span_end]
        
        # Convert to list of tuples
        tags = [(tag, tags[tag]) for tag in tags]
        
        # Map tags
        tags = [(tag[0], map_slot_value(tag[1])) for tag in tags]
        
        return tags
    
