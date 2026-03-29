from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from src.config import parse_args_llama, resolve_hf_dataset
from tqdm import tqdm
import datasets
import os
from pathlib import Path
import torch
import gc
import json
import re
import pandas as pd


def build_prompt(question):
    return f"""
You are an expert at decomposing complex questions into smaller, atomic subquestions.
If the question can't be decomposed into smaller questions, leave it as it is.

Decompose the following question into a list of simpler subquestions that:
- Are atomic (addressing only one piece of information at a time).
- Are logically ordered.
- Have access to answers from previous subquestions
- Cover all necessary aspects of the original question
- Are not yes/no questions (each must request a specific entity, not a confirmation).
- Can be answered with a single entity (e.g., a person's name, a city, a color).
- Lead to the answer in the last subquestion (the final subquestion's answer should be the answer to the original question).

You must strictly format your answer as a valid JSON array ; do NOT include explanations or reasoning.

### Examples:
Input: What is the capital of the country which exports the most honey ?
Output:
["Which country exports the most honey ?", "What is the capital of that country ?"]

Input: What sports team does Michael's best friend support ?
Output:
["Who is Michael's best friend ?", "What sports team does he support ?"]

Input: What fruits grow in the hottest countries from the largest continent in the world ?
Output:
["What is the largest continent in the world ?", "What countries are most hot on this continent ?", "What fruits grow in those countries ?"]

Input: How old is Obama ?
Output:
["How old is Obama ?"]

Now decompose the following question in the same JSON format:

"{question}"

Output:
"""

def clean_output(text):
    text = re.sub(r"</.*?>", "", text).strip()  # remove HTML tags
    print(f"Raw model output: {text}")
    try:
        parsed = json.loads(text)
        
        # check if the output is a dictionary
        if isinstance(parsed, dict) and 'subquestions' in parsed:
            subquestions = parsed['subquestions']
            # ensure that subquestions is a list of strings
            if isinstance(subquestions, list) and all(isinstance(q, str) for q in subquestions):
                return subquestions
            else:
                raise ValueError(f"Expected list of strings in 'subquestions' but got: {type(subquestions)}")
        
        # if the output is already a list of strings, return it
        elif isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return parsed
        
        else:
            raise ValueError(f"Expected list of strings or dict with key 'subquestions' but got: {type(parsed)}")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        
        with open(f"bad_subs.log", "a") as f:
            f.write(f"\n--- Malformed Output ---\n{text}\n")
        return []




def decompose_question(llm, sampling_params, question, counter):
    prompt = build_prompt(question)
    
    try:
        outputs = llm.generate(prompt, sampling_params)
        text = outputs[0].outputs[0].text.strip()
        clean = clean_output(text)
        if not clean:
            counter += 1
        return clean, counter
    except Exception as e:
        print(f"Error during generation: {e}")
        return [], counter + 1
    
def save_as_multiple_parquet(path, df, chunk_size=10000):
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
    
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        chunk_file_path = Path(path) / f"dataset_chunk_{i}.parquet"
        chunk.to_parquet(str(chunk_file_path), engine='pyarrow', index=False)
        print(f"Saved chunk {i} to {chunk_file_path}")



def generate(args):
    output_path = Path(args.decomp_datasets_dir) / args.dataset

    os.makedirs(output_path, exist_ok=True)

    #Load the CWQ dataset
    counter = 0

    dataset = load_dataset(resolve_hf_dataset(args.dataset, args.datasets_dir))
    dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])


    print(dataset[0])

    #Load the model 

    MODEL_NAME = os.getenv("DECOMPOSER_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    max_model_len=8192,
    dtype="bfloat16")

    class SubQuestion(BaseModel):
        subquestions: list[str]

    json_schema = SubQuestion.model_json_schema()

    guided_decoding_params = GuidedDecodingParams(json=json_schema)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=200,
        guided_decoding=guided_decoding_params
    )

    #Create the dataset of subquestions

    subquestions_dataset = []

    for example in tqdm(dataset, desc="Decomposing Questions"):
        subquestions, counter = decompose_question(llm=llm,question=example["question"], sampling_params=sampling_params, counter=counter)
        subquestions_dataset.append({
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "q_entity": example["q_entity"],
            "a_entity": example["a_entity"],
            "graph": example["graph"],
            "subquestions": subquestions
        })
        print(f"Counter : {counter} empty subquestions")
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(subquestions_dataset)

    save_as_multiple_parquet(output_path, df, chunk_size=2000)  # adjust chunk_size as needed

    print("Success!")


if __name__ == "__main__":
    cli_args = parse_args_llama()

    generate(cli_args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    gc.collect()
