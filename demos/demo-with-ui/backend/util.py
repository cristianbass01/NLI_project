import os
import torch
import spacy


_directory_path = os.path.dirname(os.path.abspath(__file__))
models_base_directory = os.path.join(_directory_path, 'models')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

spacy_nlp = spacy.load("en_core_web_lg")

class ModelDictionary:
    """
    models_load_functions: dict of model_id -> function that returns a model
    """
    def __init__(self, models_load_functions):
        self.models = {}
        self.models_load_functions = models_load_functions

    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        else:
            self._load_model(key)
            return self.models[key]

    def _load_model(self, model_id):
        if model_id in self.models_load_functions:
            self.models[model_id] = self.models_load_functions[model_id]()
            return
        raise f'Tried to load unknown model with id "{model_id}"'
