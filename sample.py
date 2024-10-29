#############################################

# ======== sample.py ========
# This script controls the whole pipeline of sampling


#############################################

from dataProcess import *
from prompt import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def main():
    # load datasets csv and question array
    print("==== starting ====")
    print("1. load dataset")
    loadDatasets()
    print("2. load question sets")
    questions = loadQAsets()
    print(questions)


    # create prompt
    print("3. load csvs into prompt")
    loadCsvDs()
    prompt = createPrompt(questions=questions)


    # submit prompt and receive answer
    result = process(prompt=prompt)

    # format results into a txt file with 1) question and 2) generated answer
    


    # evaluate answers

def process(prompt):
    # load  tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b", device_map="auto")

    # tokenize prompt
    prompt = prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # submit prompt and generate result
    outputs = model.generate(**inputs, max_length=500, do_sample=True)

    # decode generated response form model
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    # parse response from JSON 
    try:
        result = json.loads(response)
        print("Answer:", result.get("answer"))
        print("Columns Used:", result.get("columns_used"))
        print("Explanation:", result.get("explanation"))
    except json.JSONDecodeError:
        print("Response is not in JSON format:", response)

    return result


if __name__ == "__main__":
    main()