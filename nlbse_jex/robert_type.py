from transformers import AutoTokenizer

class RoBERTa_models():

    def __init__(self, model):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model) # TODO tokenizer is not loaded properly

    def __str__(self):
        return "hello roberta"
