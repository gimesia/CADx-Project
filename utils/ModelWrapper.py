from torch import nn

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Ensure this method matches your model's prediction interface
        return self.model(input_data)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)