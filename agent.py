import slots.slot_filler as SF

SF_MODEL_NAME = 'roberta-base'
SF_MODEL_PATH = 'slots/saved_models'

SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, cuda = True)

print(SF_model.tag_slots('Do not care how many people'))


class Agent():
    def __init__(self, intent_use_history = False, slot_use_history = False):
        # TODO: add intent predictor
        # TODO: keep only slots that match intent (I think so, idk)
        self.SF_model = SF.SlotFiller(SF_MODEL_PATH, SF_MODEL_NAME, cuda = True)
        self.utterance_history = []
        self.intent_use_history = intent_use_history
        self.slot_use_history = slot_use_history
    
    def respond(self, input):
        pass