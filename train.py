from textgenrnn import textgenrnn
import argparse

def generateText(file):
    textgen = textgenrnn()
    textgen.train_from_file(file, num_epochs=1)
    textgen.generate(1, temperature=1.0)

ap = argparse.ArgumentParser()
ap.add_argument("-data", default="./data/test.txt")
args = vars(ap.parse_args())
data = args["data"]

generateText(data)
