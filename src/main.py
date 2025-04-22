 import json
import pandas as pd
import torch
from huggingface_hub import login
from transformers import AutoTokenizer
import configparser
from datasets import Dataset
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np
from transformers import AutoModelForSeq2SeqLM


def main():
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Step 1: Load dataset
    all_data = []
    with open("datasets/datasets.jsonl") as input_file:
        for each_line in input_file:
            all_data.append(json.loads(each_line))

    # Step 2: Extract useful fields
    file_name = []
    function_name = []
    version_data = []
    diff_code = []
    diff_docstring = []
    code = []
    docstring = []

    for each_row in all_data:
        file_name.append(each_row['file'])
        function_name.append(each_row['function'])

        latest_version = 0
        latest_version_index = 0
        for i, each_version in enumerate(each_row["version_data"]):
            if int(list(each_version.keys())[0].split("v")[1]) > latest_version:
                latest_version = int(list(each_version.keys())[0].split("v")[1])
                latest_version_index = i

        version_data.append(each_row["version_data"][latest_version_index][f"v{latest_version}"])
        code.append(each_row["version_data"][latest_version_index].get("code"))
        docstring.append(each_row["version_data"][latest_version_index].get("docstring"))
        diff_code.append(each_row["diff_code"])
        diff_docstring.append(each_row["diff_docstring"])

    # Step 3: Create DataFrame
    dataframe_dict = {
        "file": file_name,
        "function": function_name,
        "version": version_data,
        "code": code,
        "docstring": docstring
    }

    df = pd.DataFrame(dataframe_dict)

    df = df[["file", "function", "code", "docstring"]]  # Drop dict columns

    print("Checking duplicates and nulls...")
    print("Duplicates:", df.duplicated().sum())
    print("Nulls:\n", df.isnull().sum())

    # Step 4: Prepare input-output
    df["input_text"] = df["code"]
    df["target_text"] = df["docstring"]

    print(df["input_text"].isnull().sum())
    print(df["target_text"].isnull().sum())

    # Step 5: Setup tokenizer
    config = configparser.ConfigParser()
    config.read('config.ini')
    huggingface_token = config["HuggingFace"]["token"]

    print("Logging to Hugging Face")
    login(token=huggingface_token)
    print("Log in successful")

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", token=huggingface_token)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small", token=huggingface_token)

    tokenized_inputs = df["input_text"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True))
    tokenized_outputs = df["target_text"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True))
    print("Tokenized input and output")

    df["input_ids"] = tokenized_inputs.apply(lambda x: x["input_ids"])
    df["attention_mask"] = tokenized_inputs.apply(lambda x: x["attention_mask"])
    df["labels"] = tokenized_outputs.apply(lambda x: x["input_ids"])

    # Step 6: Prepare datasets
    datasets = Dataset.from_pandas(df)

    train_dataset = datasets.shuffle(seed=42).select([i for i in range(int(0.8 * len(datasets)))])
    val_dataset = datasets.select([i for i in range(int(0.8 * len(datasets)), len(datasets))])

    # Step 7: Load model
    # model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

    # Step 8: Setup training args
    training_args = TrainingArguments(
        output_dir="./results",
        eval_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10
    )

    # Step 9: Metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.argmax(preds, axis=-1)  # <- FIX added
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # Replace -100 with padding token

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Calculate BLEU and METEOR manually
        bleu_scores = []
        meteor_scores = []

        for ref, pred in zip(decoded_labels, decoded_preds):
            reference_tokens = ref.split()
            predicted_tokens = pred.split()

            bleu = sentence_bleu([reference_tokens], predicted_tokens) if reference_tokens else 0.0
            meteor = meteor_score([reference_tokens], predicted_tokens) if reference_tokens else 0.0

            bleu_scores.append(bleu)
            meteor_scores.append(meteor)

        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        return {
            'bleu': avg_bleu,
            'meteor': avg_meteor,
        }

   
    # Step 10: Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Step 11: Train and Evaluate
    trainer.train()
    results = trainer.evaluate()
    print(results)

    # Save model
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

    print("Model saved!")

    # ===============================
    # ✨ Step 12: BLEU + METEOR scoring
    # ===============================

    print("Calculating BLEU and METEOR scores...")

    bleu_scores = []
    meteor_scores = []

    input_texts = df["input_text"].tolist()
    target_texts = df["target_text"].tolist()

    generated_texts = []

    for input_code in input_texts:
        inputs = tokenizer(input_code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
        predicted_docstring = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(predicted_docstring.strip())

    for ref, pred in zip(target_texts, generated_texts):
        reference_tokens = ref.split()
        predicted_tokens = pred.split()

        bleu = sentence_bleu([reference_tokens], predicted_tokens) if reference_tokens else 0.0
        meteor = meteor_score([reference_tokens], predicted_tokens) if reference_tokens else 0.0

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    print(f"Average METEOR Score: {avg_meteor:.4f}")

if __name__ == "__main__":
    main()

