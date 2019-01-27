from textgenrnn import textgenrnn
from os import path
import argparse
import requests
import random
import sched
import json
import time
import glob

s = sched.scheduler(time.time, time.sleep)

fileDict = {
    "pt0" : "./data/Ultra__Travolta.txt",
    "pt1" : "./data/plastic-bubble.txt",
    "pt2" : "./data/pulp-ezekiel.txt",
    "pt3" : "./data/battlefield-cat.txt",
    "pt4" : "./data/Ultra__Travolta.txt",
    "pt5" : "./data/hairspray-sutta.txt"
}

ap = argparse.ArgumentParser()
ap.add_argument("-data")
ap.add_argument("-botkey")
ap.add_argument("-temperature", default=0.5)
ap.add_argument("-baseUrl", default="http://localhost:3000")
ap.add_argument("-loop", default=False)
ap.add_argument("-picOdds", default=0.75)
args        = vars(ap.parse_args())
data        = args["data"] if args["data"] else fileDict[args["botkey"]] if args["botkey"] else "./data/test.txt"
botkey      = args["botkey"] if args["botkey"] else "pt0"
temperature = args["temperature"]
baseUrl     = args["baseUrl"]
loop        = args["loop"]
picOdds     = float(args["picOdds"])

textgen = None

def coinFlip(bias=0.5):
    return random.random() < bias

def generateText(botkey, file):
    weights      = "{}_weights.hdf5".format(botkey)
    weights_file = weights if path.isfile(weights) else None
    vocab_file   = None
    config_file  = None
    new          = False

    # # OPTION OF TRAINING FROM ENTIRELY NEW MODEL
    # weights_file = None
    # vocab_file = None
    # config_file = None
    # new = True
    # if path.isfile(weights):
    #     weights_file = weights
    #     vocab_file = "{}_vocab.json".format(botkey)
    #     config_file = "{}_config.json".format(botkey)
    #     new = False

    textgen = textgenrnn(
        name         = botkey,
        weights_path = weights_file,
        vocab_path   = vocab_file,
        config_path  = config_file,
    )
    textgen.train_from_file(file, num_epochs=1, new_model=new)
    list = textgen.generate(1, temperature=temperature, max_gen_length=280, return_as_list=True)
    return list[0]

def submitStatus(botkey, body):
    if (not botkey):
        print("No botkey provided")
        return
    if (not body):
        print("No body provided")
        return
    data     = {"status": body}
    endpoint = "{}/status/{}".format(baseUrl, botkey)
    if (coinFlip(picOdds)):
        file  = random.choice(glob.glob("./media/{}/*.jpg".format(botkey)))
        files = {"file": ("media", open(file, "rb"), "image/jpeg")}
        return requests.post(endpoint, files=files, data=data)
    return requests.post(endpoint, json=data)

def makeBot(botkey, file):
    status = generateText(botkey, file)
    submitStatus(botkey, status)
    textgen = None
    return

def loopBots():
    idx = 0
    while idx < len(botkeys):
        key     = "pt" + str(idx)
        file    = fileDict[key]
        status  = generateText(key, file)
        submitStatus(key, status)
        textgen = None
        idx += 1
    s.enter(7200, 1, loopBots)

if loop:
    s.enter(0, 1, loopBots)
    s.run()
else:
    makeBot(botkey, data)
