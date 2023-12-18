import numpy as np

class DictModel():
    def __init__(self):
        self.model_dict = {}
    
    def fit(self, X, y):
        self.num_classes = y.shape[1]
        self.model_dict = {tuple(train_in) : train_out for train_in, train_out in zip(X, y)}
        
    
    def predict(self, X):
        predicted_output = []
        for test_in in X:
            if tuple(test_in) not in self.model_dict:
                predicted_output.append(np.zeros(self.num_classes))
            else:
                predicted_output.append(self.model_dict[tuple(test_in)])
        return np.array(predicted_output)