#############################################

# ======== dataProcess.py ========
# This script loads both the dataset and question 
# set into seperate csv files stored in "csvDb"


#############################################

from datasets import load_dataset
import pandas as pd

all_qa = ""
questions = []

def loadDatasets():

    # load datasets 
    # dataBenchLite = load_dataset("cardiffnlp/databench", "lite")
    global all_qa
    all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")
    # semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")
    # semeval_dev_qa = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
    print("==== dataBench loaded ====")

    # transfer to pandas
    ds_id = all_qa['dataset'][0]
    # print(ds_id)
    pd_ds_id = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{ds_id}/all.parquet")
    # pd_dataBenchLite = pd.DataFrame(dataBenchLite['dev'])
    print("==== transformed to pandas ====")

    # format to csv
    csv_ds_id = pd_ds_id.to_csv(index=False)
    print("==== formatted to csv ====")

    # create the markdown block with csv
    markdown_ds_id = f"```csv\n{csv_ds_id}\n```"

    # download csv file for testing
    fileName = "dataBenchCsv"
    filePath = "./csvDb/" + fileName
    with open(filePath, 'w') as file:
        file.write(markdown_ds_id)
    print("==== CSV file downloaded in " + filePath + " ====")
    print("#################### Dataset from table: " + ds_id + " ####################")
    print(pd_ds_id)

def loadQAsets():
    # qs_id = all_qa
    # print(qs_id)
    pd_qs = pd.DataFrame(all_qa)
    pd_qs_id = pd_qs.head(5)
    global questions 
    questions = pd_qs_id['question']
    # print(questions)
    print("==== transfer to pandas ====")

    csv_qs_id = pd_qs_id.to_csv(index=False)
    print("==== convert to csv ====")
    # create the markdown block with csv
    markdown_qs_id = f"```csv\n{csv_qs_id}\n```"

    fileName = "questionSetCsv"
    filePath = "./csvDb/" + fileName
    with open(filePath, 'w') as file:
        file.write(markdown_qs_id)
    print("==== CSV file downloaded in " + filePath + " ====")
    print("#################### question set ####################")
    print(pd_qs_id)


    return questions


def main():
    print("==== starting ====")
    print("1. load dataset")
    loadDatasets()
    print("2. load question sets")
    loadQAsets()

if __name__ == "__main__":
    main()