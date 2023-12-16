from flask import Flask
from flask_restful import Resource, Api, reqparse
from collections import defaultdict
from task_1 import predict_domain_and_dialog_act

app = Flask(__name__)
api = Api(app)

# Your existing data structures
prediction_histories = defaultdict(list)
message_histories = defaultdict(list)

# Request parser setup


class PredictDomainAndDialogAct(Resource):
    def __init__(self):
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('conversation-id', required=True, help="conversation-id cannot be blank")
        parser.add_argument('message', required=True, help="message cannot be blank")
        parser.add_argument('model-id', required=True, help="model-id cannot be blank")
        self.parser = parser

    def get(self):
        args = self.parser.parse_args()
        conversation_id = args['conversation-id']
        message = args['message']
        model_id = args['model-id']

        historical_messages = message_histories[conversation_id]
        historical_predictions = prediction_histories[conversation_id]

        prediction = predict_domain_and_dialog_act(message, historical_messages, historical_predictions, model_id)
        prediction_histories[conversation_id].append(prediction)

        return {'prediction': predict_domain_and_dialog_act(message, historical_messages, historical_predictions, model_id)}



# Add the resource to the API
api.add_resource(PredictDomainAndDialogAct, '/task-1/predict-domain-and-dialog-act')

if __name__ == '__main__':
    app.run(debug=True)
