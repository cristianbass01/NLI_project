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

TRANSFORMER_MODEL_NAME = 'roberta-base'
MODEL_PATH = 'saved_models'

def load_intent_classification_model():
    pass

def load_slot_filling_model(model_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # !!! Make sure to change this as well if tags are changed
    possible_tags = ['B-area', 'B-bookday', 'B-bookpeople', 'B-bookstay', 'B-booktime', 'B-food', 'B-name', 'B-pricerange', 'B-stars', 'B-type', 'I-area', 'I-bookday', 'I-bookpeople', 'I-booktime', 'I-food', 'I-name', 'I-pricerange', 'I-type', 'O']
    transformer = RobertaForTokenClassification.from_pretrained(os.path.join(model_path, 'SF_' + model_name), num_labels = len(possible_tags))
    transformer.eval()
    
    return tokenizer, transformer, possible_tags

tokenizer, transformer, possible_tags = load_slot_filling_model(MODEL_PATH, TRANSFORMER_MODEL_NAME)

def tag_slots(utterance):
    tokenized = tokenizer(utterance, return_tensors = 'pt', padding = 'max_length')
    out = transformer.forward(input_ids = tokenized.input_ids, attention_mask = tokenized.attention_mask)
    pred = torch.argmax(out.logits, dim = 2).squeeze()
    # tag name, value
    tags = []
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
            tags.append((cur_type, utterance[span_start : span_end]))
            
            
        # if tokenized.token_to_word(i) is not None and tokenized.token_to_word(i) != last_pred_word:
        #     last_pred_word = tokenized.token_to_word(i)
        #     word_tags.append(possible_tags[pred[i]])
    
    return tags

print(tag_slots('We love doing NLI!'))