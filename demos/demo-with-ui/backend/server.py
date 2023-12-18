from flask import Flask
from flask_restful import Resource, Api, reqparse
from collections import defaultdict
from task_1 import predict_domain_and_dialog_act
from task_2 import predict_semantic_frame_slot_filling
from full_pipeline import process_user_utterance

app = Flask(__name__)
api = Api(app)

# Store the historical messages and predictions for each conversation
message_histories = defaultdict(list)
domain_and_dialog_acts_histories = defaultdict(list)
slot_values_histories = defaultdict(list)
slot_questions_histories = defaultdict(list)


# Full Pipeline
class FullPipeline(Resource):
    def __init__(self):
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('message', required=True, help="message cannot be blank")
        self.parser = parser

    def post(self):
        args = self.parser.parse_args()
        message = args['message']
        return process_user_utterance(message)





# Task 1
class PredictDomainAndDialogAct(Resource):
    def __init__(self):
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('conversation-id', required=True, help="conversation-id cannot be blank")
        parser.add_argument('message', required=True, help="message cannot be blank")
        parser.add_argument('model-id', required=True, help="model-id cannot be blank")
        self.parser = parser

    def post(self):
        args = self.parser.parse_args()
        conversation_id = args['conversation-id']
        message = args['message']
        model_id = args['model-id']

        historical_messages = message_histories[conversation_id]
        historical_domain_and_dialog_acts = domain_and_dialog_acts_histories[conversation_id]

        parsed_sentence, prediction = predict_domain_and_dialog_act(message, historical_messages, historical_domain_and_dialog_acts, model_id)
        
        # Update history
        message_histories[conversation_id].append(message)
        domain_and_dialog_acts_histories[conversation_id].append(prediction)

        return {
            'prediction': prediction,
            'message': message,
            'parsed_sentence': parsed_sentence,
            }



# Task 2
class PredictSemanticFrameSlotFilling(Resource):
    def __init__(self):
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('conversation-id', required=True, help="conversation-id cannot be blank")
        parser.add_argument('message', required=True, help="message cannot be blank")
        parser.add_argument('model-id', required=True, help="model-id cannot be blank")
        self.parser = parser

    def post(self):
        args = self.parser.parse_args()
        conversation_id = args['conversation-id']
        message = args['message']
        model_id = args['model-id']

        historical_messages = message_histories[conversation_id]
        historical_domain_and_dialog_acts = domain_and_dialog_acts_histories[conversation_id]
        historical_slot_values = slot_values_histories[conversation_id]
        historical_slot_questions = slot_questions_histories[conversation_id]

        slot_values, slot_questions = predict_semantic_frame_slot_filling(message, historical_messages, historical_domain_and_dialog_acts, historical_slot_values, historical_slot_questions, model_id)

        # Update history
        slot_values_histories[conversation_id].append(slot_values)
        slot_questions_histories[conversation_id].append(slot_questions)

        return {
            'slot_values': slot_values,
            'slot_questions': slot_questions,
            }


# Add the resources to the API
api.add_resource(PredictDomainAndDialogAct, '/task-1/predict-domain-and-dialog-act')
api.add_resource(PredictSemanticFrameSlotFilling, '/task-2/predict-semantic-frame-slot-filling')
api.add_resource(FullPipeline, '/full-pipeline')

if __name__ == '__main__':
    app.run(debug=True)
