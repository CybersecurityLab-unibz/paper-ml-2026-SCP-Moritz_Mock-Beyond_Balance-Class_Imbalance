from transformers import AutoTokenizer

class RoBERTaModels():

    def __init__(self, args):
        self.args = args
        self.model = self.args.model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model) # TODO tokenizer is not loaded properly

    def __str__(self):
        return "hello roberta"
