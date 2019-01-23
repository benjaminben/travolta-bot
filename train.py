from textgenrnn import textgenrnn
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-data", default="./data/test.txt")
args = vars(ap.parse_args())
data = args["data"]

textgen = textgenrnn()
textgen.train_from_file(data, num_epochs=20)
textgen.generate(1, temperature=1.0)
