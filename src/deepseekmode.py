import json
import pandas as pd
import requests
import configparser
from tqdm import tqdm
import time
import nltk
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')
nltk.download('omw-1.4')

def query_huggingface_api(payload, model_name, api_token, retry=3, wait_time=5):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}

    for attempt in range(retry):
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Warning: API call failed (status {response.status_code}). Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print("Error: Failed after retries.")
    return None

def main():
    # Step 1: Read JSONL file
    all_data = []
    try:
        with open("codocbench/dataset/codocbench.jsonl") as input_file:
            for each_line in input_file:
                all_data.append(json.loads(each_line))
        print(f"Successfully loaded {len(all_data)} samples from file.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if not all_data:
        print("No data loaded. Exiting...")
        return

    # Step 2: Extract latest code and docstring
    code = []
    original_docstring = []

    for each_row in all_data:
        version_data = each_row.get('version_data', [])
        if not version_data:
            continue

        latest_entry = max(version_data, key=lambda x: x.get('commit_date_time', ''))

        code_value = latest_entry.get('code', '')
        docstring_value = latest_entry.get('docstring', '')

        if code_value.strip() == "" or docstring_value.strip() == "":
            continue

        code.append(code_value)
        original_docstring.append(docstring_value)

    print(f"Loaded {len(code)} valid samples after filtering.")

    if not code:
        print("No valid data after filtering. Exiting...")
        return

    # Step 3: Create DataFrame (ONLY code and docstring)
    df = pd.DataFrame({
        "code": code,
        "original_docstring": original_docstring
    })

    df = df.head(50)  # Work only with 50 samples to avoid heavy API load
    print(f"Final dataset prepared with {len(df)} samples.")

    # Step 4: Load Huggingface API token
    config = configparser.ConfigParser()
    config.read('config.ini')

    if "HuggingFace" not in config or "token" not in config["HuggingFace"]:
        print("Huggingface token not found in config.ini. Exiting...")
        return

    huggingface_token = config["HuggingFace"]["token"]

    # Step 5: DeepSeek model setup
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

    # Step 6: Pick few-shot examples
    few_shot_df = df.sample(n=5, random_state=42)  # Pick 5 random examples
    few_shot_examples = ""
    for _, row in few_shot_df.iterrows():
        few_shot_examples += f"Code:\n{row['code']}\n\nDocstring:\n{row['original_docstring']}\n\n"

    # Step 7: Generate docstrings using DeepSeek API
    generated_docstrings = []

    print(" Starting generation using DeepSeek...")
    
    for code_sample in tqdm(df["code"]):
        input_prompt = few_shot_examples + f"""
Now generate a docstring for:

Code:
{code_sample}

Docstring:
"""

        payload = {
            "inputs": input_prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }

        response = query_huggingface_api(payload, model_name, huggingface_token)

        if response and isinstance(response, list) and "generated_text" in response[0]:
            generated_doc = response[0]["generated_text"].strip()
        else:
            generated_doc = "ERROR"

        generated_docstrings.append(generated_doc)

    df["deepseek_generated_docstring"] = generated_docstrings

    # Step 8: Save ONLY needed columns
    df_final = df[["original_docstring", "deepseek_generated_docstring"]]
    df_final.to_csv("deepseek_docstrings_with_fewshot.csv", index=False)
    print("Saved clean output to deepseek_docstrings_with_fewshot.csv!")

    # Step 9: Calculate BLEU and METEOR
    print(" Calculating BLEU and METEOR scores...")

    smooth_fn = SmoothingFunction().method2

    bleu_scores = []
    meteor_scores = []

    for ref, hyp in zip(df["original_docstring"], df["deepseek_generated_docstring"]):
        if hyp == "ERROR" or ref.strip() == "" or hyp.strip() == "":
            continue  # Skip bad samples

        reference = [ref.split()]  # BLEU needs list-of-list
        hypothesis = hyp.split()

        bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth_fn)

        reference_tokens = ref.split()     # METEOR needs pre-tokenized
        hypothesis_tokens = hyp.split()
        meteor = meteor_score([reference_tokens], hypothesis_tokens)

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)

    if bleu_scores and meteor_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        print(f"Average BLEU Score: {avg_bleu:.4f}")
        print(f"Average METEOR Score: {avg_meteor:.4f}")
    else:
        print("No valid samples for evaluation. All outputs were ERROR or empty.")

if __name__ == "__main__":
    main()
