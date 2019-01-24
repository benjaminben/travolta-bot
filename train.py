from textgenrnn import textgenrnn
import argparse
import requests
import os.path
import json

ap = argparse.ArgumentParser()
ap.add_argument("-data", default="./data/test.txt")
ap.add_argument("-baseUrl", default="http://localhost:3000")
ap.add_argument("-botkey", default="pt0")
args = vars(ap.parse_args())
data = args["data"]
botkey = args["botkey"]
baseUrl = args["baseUrl"]

textgen = None

def generateText(botkey, file):
    weights = "{}_weights.hdf5".format(botkey)
    weights_file = weights if os.path.isfile(weights) else None
    vocab_file = None
    config_file = None
    new = False

    # # OPTION OF TRAINING FROM ENTIRELY NEW MODEL
    # weights_file = None
    # vocab_file = None
    # config_file = None
    # new = True
    # if os.path.isfile(weights):
    #     weights_file = weights
    #     vocab_file = "{}_vocab.json".format(botkey)
    #     config_file = "{}_config.json".format(botkey)
    #     new = False

    textgen = textgenrnn(
        name=botkey,
        weights_path=weights_file,
        vocab_path=vocab_file,
        config_path=config_file,
    )
    textgen.train_from_file(file, num_epochs=1, new_model=new)
    list = textgen.generate(1, temperature=1.0, max_gen_length=280, return_as_list=True)
    return list[0]

def submitStatus(botkey, body):
    if (not botkey):
        print("No botkey provided")
        return
    if (not body):
        print("No body provided")
        return
    return requests.post(baseUrl + "/status/" + botkey, json={"status": body})

def makeBot(botkey, file):
    status = generateText(botkey, file)
    submitStatus(botkey, status)
    textgen = None
    return

makeBot(botkey, data)
