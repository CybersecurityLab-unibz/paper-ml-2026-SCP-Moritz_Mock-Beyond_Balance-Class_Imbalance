import types
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import read_config
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitTrainer
from datasets import Dataset, load_dataset
import numpy as np

def main():
    args = parse_args()
    if args.generate_config_hyper_parameter_search is True:
        path = "config.yml"
        generate_base_config(path)
        print(f"Example configuration file was generated under the path {path}")
        quit()

    if args.config is not None:
        args = read_config(args)

    args = read_files_weighted_loss(args)

    for wl in args.weighted_loss:
        if isinstance(wl, str):
            print(wl)
        if isinstance(wl, types.FunctionType):
            wl("moritz")


    print(args)

    ########################### Single task ###########################
    '''
    train_dataset = Dataset.from_dict({
        "text": [
            "I love this!", "That's great!", "Amazing experience.",
            "I hate it", "That's terrible", "Worst ever."
        ],
        "label": [1, 1, 1, 0, 0, 0]
    })

    # Step 2: Load a SetFit model (Sentence Transformer under the hood)
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    # Step 2.5
    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Step 3: Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset
    )

    trainer.train()

    # Step 4: Inference
    preds = model.predict(["I really enjoy this", "It was awful"])
    print(preds)  # e.g., [1, 0]
    '''

    ########################### Multi task ###########################
    # ---------- Step 1: Multi-output dataset ----------
    # Each sample has two labels: [sentiment, emotion]
    # ---------- Step 1: Multi-output dataset ----------
    langs = ['java', 'python', 'pharo']
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }
    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')


    dataset = Dataset.from_dict({
        "text": [
            "I love this!", "That's great!", "Amazing experience.",
            "I hate it", "That's terrible", "Worst ever."
        ],
        "label": [
            [1, 1], [1, 1], [1, 1],
            [0, 2], [0, 2], [0, 2]
        ]
    })

    # Convert labels to NumPy array
    dataset = dataset.map(lambda x: {"label": np.array(x["label"], dtype=int)})

    # Model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_target_strategy="multi-output"
    )

    lan = "java"

    # Trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=ds[f'{lan}_train'],
        eval_dataset=ds[f'{lan}_test'],
        column_mapping={"combo": "text", "labels": "label"},
        num_epochs=5 if lan == 'java' else 10,
        batch_size=32,
    )

    trainer.train()

    # Inference
    preds = model.predict(["I really enjoy this", "It was awful"])
    print(preds)

if __name__ == "__main__":
    main()
