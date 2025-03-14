# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

print("start...")
tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/StructLM-7B")
model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/StructLM-7B")

print("setting up input...")
input_string = """[INST] <<SYS>>
    You are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.
    <</SYS>>

    Use the information in the following table to solve the problem, choose between the choices if they are provided. table:

    col : day | kilometers row 1 : tuesday | 0 row 2 : wednesday | 0 row 3 : thursday | 4 row 4 : friday | 0 row 5 : saturday | 0


    question:

    Allie kept track of how many kilometers she walked during the past 5 days. What is the range of the numbers? [/INST]
    """

# input_ids = input_ids.to('cpu')
print(input_string)
inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cpu")
print("generating outputs...")
outputs = model.generate(inputs, max_length=200)

print(tokenizer.decode(outputs[0]))

