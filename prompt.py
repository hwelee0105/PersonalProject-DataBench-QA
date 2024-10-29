#############################################

# ======== prompt.py ========
# This script loads both the dataset and question 
# set into seperate csv files stored in "csvDb"


#############################################

from dataProcess import questions
qaCsv = None
dsCsv = None

def loadCsvQs():
    filePath = "./csvDb/questionSetCsv"
    with open(filePath, 'r') as file:
        global qaCsv
        qaCsv = file.read()

def loadCsvDs():
    filePath = "./csvDb/dataBenchCsv"
    with open(filePath, 'r') as file:
        global dsCsv
        dsCsv = file.read()

def createPrompt(questions):
    question = questions[0]
    prompt = 'You are an assistant tasked with answering the questions asked of a given CSV in JSON format. You must answer in a single JSON with three fields: * "answer": answer using information from the provided CSV only. * "columns_used": list of columns from the CSV used to get the answer. * "explanation": A short explanation on why you gave that answer. Requirements: * Only respond with the JSON. In the following CSV {} USER: {} ASSISTANT: "answer":'.format(dsCsv, question)
    
    filePath = "./prompt.txt"
    with open(filePath, 'w') as file:
        file.write(prompt)

    return prompt

def main():
    loadCsvDs()

if __name__ == "__main__":
    main()