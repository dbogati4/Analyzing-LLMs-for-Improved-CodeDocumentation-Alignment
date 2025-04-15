import json
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer
import configparser
from datasets import Dataset
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments



def main():
    # step 1: open and read the jsonl file
    all_data = []
    with open("datasets/datasets.jsonl") as input_file:
        for each_line in input_file:
            all_data.append(json.loads(each_line))
    
    # step 2: extract and store useful fields 
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

    # store useful fields in dataframe
    dataframe_dict = {
        "file": file_name,
        "function": function_name,
        "version": version_data,
        "code": code,
        "docstring": docstring
    } 
    
    df = pd.DataFrame(dataframe_dict)
    
    # converting to string to check the duplicate value
    df["version"] = df["version"].apply(str)
    count_duplicates = df.duplicated().sum()
    print(count_duplicates)  # It verifies that there is no duplicate value in datasets
    
    # count the number of null values
    count_null = df.isnull().sum()
    print(count_null)  # It verifies the datasets is pretty clean as there is no null value as well
    
    # step 5: prepare input-output pairs for the model
    df["input_text"] = df["code"]
    print(df["input_text"].isnull().sum())  # counting null value
    
    df["target_text"] = df["docstring"]
    print(df["target_text"].isnull().sum())
    
    # step 6: Setting up the tokenizer
    config = configparser.ConfigParser()
    config.read('config.ini')
    huggingface_token = config["HuggingFace"]["token"]
    
    print("Logging to Hugging Face")
    login(token=huggingface_token)
    print("Log in successful")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", token=huggingface_token)  # this is using CODEBeRT model

    tokenized_inputs = df["input_text"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=512))
    tokenized_outputs = df["target_text"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=128))
    print("Tokenized input and output files")
    
    df["input_ids"] = tokenized_inputs.apply(lambda x: x["input_ids"])
    df["attention_mask"] = tokenized_inputs.apply(lambda x: x["attention_mask"])  # to indicate the actual content and the padded content
    df["labels"] = tokenized_outputs.apply(lambda x: x["input_ids"])

    # Step 7: Prepare the datasets for training
    datasets = Dataset.from_pandas(df)
    
    # Split the dataset into 80-20% training and validation datasets
    train_dataset = datasets.shuffle(seed=42).select([i for i in range(int(0.8 * len(datasets)))])
    val_dataset = datasets.select([i for i in range(int(0.8 * len(datasets)), len(datasets))])
  
    # define the model
    model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")
    
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= val_dataset
    )
    
    trainer.train()
    
    trainer.evaluate()
    
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")


if __name__ == "__main__":
    main()
    