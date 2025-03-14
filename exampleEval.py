import pandas as pd
import subprocess
import shlex
import zipfile
import argparse

from databench_eval import Runner, Evaluator, utils

from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import torch

parser = argparse.ArgumentParser(description='Sampling for dataBenchQA')
parser.add_argument('--eval', action='store_true', help='run evaluation on the entire dataset')
parser.add_argument('--sample', action='store_true', help='generate sample on a single question')
args = parser.parse_args()


# ===========================================================================================
#                                    Utils functions 
# ===========================================================================================


model_name = 'deepseek-ai/deepseek-llm-7b-base'


# from llama_cpp import Llama

# model = Llama(model_path="./models/stable-code-3b.Q4_K_M.gguf")
# model = Llama.from_pretrained(
# 	repo_id="TheBloke/stable-code-3b-GGUF",
# 	filename="stable-code-3b.Q3_K_M.gguf",
# )

# this makes use of https://huggingface.co/TheBloke/stable-code-3b-GGUF
# and https://github.com/ggerganov/llama.cpp

# Uses Flan-UL2
offload_folder = "offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)
# os.makedirs(offload_folder, exist_ok=True)

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-ul2", 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder=offload_folder,
    )                                                                 
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")


# Uses StructLM
# tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/StructLM-7B")
# model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/StructLM-7B")

# Uses TableGPT2-7B
# tokenizer = AutoTokenizer.from_pretrained("tablegpt/TableGPT2-7B")
# model = AutoModelForCausalLM.from_pretrained("tablegpt/TableGPT2-7B", torch_dtype="auto", device_map="auto")

# Uses DeepSeek-7b
# tokenizer = AutoTokenizer.from_pretrained("tablegpt/TableGPT2-7B")
# model = AutoModelForCausalLM.from_pretrained("tablegpt/TableGPT2-7B", torch_dtype="auto", device_map="auto")

# For General Use:
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, 
#                                              torch_dtype=torch.bfloat16, 
#                                              device_map="auto", 
#                                              offload_folder=offload_folder
#                                              )

# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id


# second try this makes use of https://huggingface.co/JOSHMT0744/TableLlama-Q4_K_M-GGUF
def call_gguf_model(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True) #.input_ids.to(model.device)
    # inputs = tokenizer(prompts, return_tensors="pt") #.input_ids.to("cpu")
    results = model.generate(**inputs.to(model.device), max_new_tokens=100) # ,max_length=700

    # For TableGPT2-7B
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompts},
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # print(text)
    # inputs = tokenizer([messages], return_tensors="pt").to(model.device)

    # results = model.generate(**inputs, max_new_tokens=512)
    # results = [
    #     output_ids[len(input_ids) :]
    #     for input_ids, output_ids in zip(inputs.input_ids, results)
    # ]


    # results = []
    # i = 0
    # for p in prompts:
    #     # print("prompt is.... \n", p)
    #     # if i < 75:
    #     #     continue
    #     # i = i +1
    #     escaped = p.replace('"', '\\"') 
    #     # truncated_prompt = escaped[:350]
    #     cmd = f'llama-cli --hf-repo "JOSHMT0744/TableLlama-Q4_K_M-GGUF" --hf-file tablellama-q4_k_m.gguf -p "{escaped}" -c 1024 -n 128'
    #     # cmd = f'llama-cli --hf-repo "TheBloke/stable-code-3b-GGUF" --hf-file stable-code-3b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128'
    #     # cmd = f'llama-cli -m ./models/stable-code-3b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128'
    #     args = shlex.split(cmd)
    #     try:
    #         result = subprocess.run(args, capture_output=True, text=True, check=True)
    #         results.append(result.stdout)
    #         # response = model(truncated_prompt, max_tokens=128, temperature=0.7)  # Adjust parameters as needed
    #         # results.append(response["choices"][0]["text"])
    #         # print("result is ================ \n", result)
    #     except Exception as e:
    #         print("ERROR ================ \n", e.stderr)
    #         results.append(f"__CODE_GEN_ERROR__: {e.stderr}")

    return results

def format_dataframe(df):
    """
    Format a DataFrame into a specific string format:
    - [TAB] indicates the beginning of the table.
    - [SEP] separates each row.
    - "|" separates each cell within a row.
    """
    formatted_table = "[TAB]\n"  # Start with the [TAB] tag

    # Format the column names
    columns = "|".join(df.columns.astype(str))
    formatted_table += columns + "[SEP]"

    rows = []
    
    for _, row in df.iterrows():
        # Convert each row into a "|" separated string
        formatted_row = "|".join(map(str, row.values))
        rows.append(formatted_row)
    
    # Combine all rows with [SEP] as the separator
    formatted_table += "[SEP]".join(rows)
    
    return formatted_table

def example_generator(row: dict) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = utils.load_table(dataset)
    # df.to_csv(index=False)
    # return f"""Answer the following quesion: {question} data: {df}"""
    # ----------- for Tablellama -------------
    # df_formatted = format_dataframe(df)
    # prompts = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    # ### Instruction: 
    # This is a hierachical table question answering task. The goal for this task is to answer the given qeustion based on the given table. The table might be hierachical.
    
    # ### Input:
    # {df}

    # ### Question:
    # {question}

    # ### Response:"""
    # ----------- for original model used -------------
    # prompts = f"""
    # # TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
    # def example(df: pd.DataFrame) -> int:
    #     df.columns=["A"]
    #     return df.shape[0]

    # # TODO: complete the following function in one line. It should give the answer to: {question}
    # def answer(df: pd.DataFrame) -> {row["type"]}:
    #     df.columns = {list(df.columns)}
    #     return"""
    # ----------- for flan_UL2 -------------
    # df_formatted = format_dataframe(df)
    prompts = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. The response must strictly adhere to the output format, if specified.
    ### Instruction: 
    This is a hierachical table question answering task. The goal for this task is to answer the given qeustion based on the given table. The table might be hierachical.
    
    ### Input:
    {df}

    ### Question:
    {question}

    ### Output format:
    {row["type"]}

    ### Response:
     """

    # ----------- for StructLM -------------
    # df_formatted = format_dataframe(df)
    # return f"""[INST] <<SYS>>
    # You are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.
    # <</SYS>>

    # Use the information in the following table to solve the problem, choose between the choices if they are provided. table:

    # {df}


    # question:

    # {question} [/INST]
    # """

 # ----------- for TableGPT2-7B -------------
    # df = df.map(lambda x: str(x) if isinstance(x, list) else x)
    # df_info = df.head(5).to_string(index=False)

    # prompts = f"""Given access to several pandas dataframes, write the Python code to answer the user's question.

    # /*
    # "df.head(5).to_string(index=False)" as follows:
    # {df_info}
    # */

    # Question: {question}
    # """

    # if isinstance(prompts, list):
    #     prompts = " ".join(prompts)

    # ----------- for DeepSeek-7b -------------
    # prompts = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    # ### Instruction: 
    # This is a hierachical table question answering task. The goal for this task is to answer the given qeustion based on the given table. The table might be hierachical.

    # ### Input:
    # {df}

    # ### Question:
    # {question}

    # ### Response:"""

    return prompts

def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        response = tokenizer.decode(response[0])
        # response = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        # print("response: ", response)
#         lead = """
# def answer(df):
#     return """
#         exec(
#             "global ans\n"
#             + lead
#             + response.split("return")[2]
#             .split("\n")[0]
#             .strip()
#             .replace("[end of text]", "")
#             + f"\nans = answer(df)"
#         )
#         # no true result is > 1 line atm, needs 1 line for txt format
#         return ans.split("\n")[0] if "\n" in str(ans) else ans
        # print(response.split(">")[1].split("<")[0].strip())
        return response.split(">")[1].split("<")[0].strip()
    except Exception as e:
        return f"__CODE_ERROR__: {e}"

# ===========================================================================================
#                                    Sampling functions 
# ===========================================================================================
def evaluate():
    qa = utils.load_qa(name="semeval", split="dev")
    evaluator = Evaluator(qa=qa)

    # runner = Runner(
    #     model_call=call_gguf_model,
    #     prompt_generator=example_generator,
    #     postprocess=lambda response, dataset: example_postprocess(
    #         response, dataset, utils.load_table
    #     ),
    #     qa=qa,
    #     batch_size=10,
    # )

    runner_lite = Runner(
        model_call=call_gguf_model,
        prompt_generator=example_generator,
        postprocess=lambda response, dataset: example_postprocess(
            response, dataset, utils.load_sample
        ),
        qa=qa,
        batch_size=10,
    )

    # responses = runner.run(save="predictions.txt")
    responses_lite = runner_lite.run(save="predictions_lite.txt")
    # responses = "predictions.txt"
    # print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.15
    print(
        f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}"
    )  # ~0.07


    # with zipfile.ZipFile("Archive.zip", "w") as zipf:
        # zipf.write("predictions.txt")
        # zipf.write("predictions_lite.txt")

    print("Created Archive.zip containing predictions.txt and predictions_lite.txt")

def sampling():
    tableID = "060_Bakery"
    question = "List the top 3 items most frequently bought in the morning."  
    answerType = "list[category]"
    sampleAnswer = "['Coffee', 'Bread', 'Pastry']"

    input = {
        "dataset": tableID,
        "question": question,
        "type": answerType,
        "sample_answer": sampleAnswer, 
    }

    # Step 1: Generate the prompt
    generated_prompt = example_generator(input)
    print("Generated Prompt:\n", generated_prompt)

    # Step 2: Call the model
    responses = call_gguf_model(generated_prompt) 

    # Flan-UL2
    # inputs = tokenizer(generated_prompt, return_tensors="pt").input_ids.to("cuda")
    # responses = model.generate(inputs, max_length=200)

    # StructLM
    # input_string = """[INST] <<SYS>>
    # You are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.
    # <</SYS>>

    # Use the information in the following table to solve the problem, choose between the choices if they are provided. table:

    # col : day | kilometers row 1 : tuesday | 0 row 2 : wednesday | 0 row 3 : thursday | 4 row 4 : friday | 0 row 5 : saturday | 0


    # question:

    # Allie kept track of how many kilometers she walked during the past 5 days. What is the range of the numbers? [/INST]
    # """
    # inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
    # responses = model.generate(inputs, max_length=200)

    # print("Model Response:\n", responses[0])
    # response = tokenizer.decode(responses[0])
    response = tokenizer.decode(responses[0], skip_special_tokens=True) # tokenizer.batch_decode(responses, skip_special_tokens=True)[0]
    print("Model Response:\n", response)

    # Step 3: Post-process the response
    processed_response = example_postprocess(responses, tableID, utils.load_table)
    print("Processed Answer:\n", processed_response)



# ===========================================================================================
#                                       Main 
# ===========================================================================================
if args.eval:
    evaluate()
elif args.sample:
    sampling()
