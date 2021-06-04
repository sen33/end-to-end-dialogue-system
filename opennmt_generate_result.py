import json
from collections import defaultdict
import argparse
from argparse import ArgumentParser

parser = ArgumentParser(description='Generate result in certain format.')
parser.add_argument("--model_id", type=str, help="ID of model", required = True)

args = parser.parse_args()

file_src = open('data/src-test.txt', 'r')
file_pred = open(args.model_id + '/pred-test.txt', 'r')

src = file_src.read().split('\n')
pred = file_pred.read().split('\n')

counter = -1
output = defaultdict(list)

for i, pred_sentence in enumerate(pred):
	if (not '<sys>' in src[i]):
		counter += 1
	# 	temp = {}
	# 	temp['spk'] = 'SYS'
	# 	temp['text'] = pred_sentence
	# 	output[str(counter)].append(temp)
	# else:
	temp = {}
	temp['spk'] = 'SYS'
	temp['text'] = pred_sentence
	output[str(counter)].append({"spk" : "SYS", 'text':pred_sentence})

with open(args.model_id + '/result.json', 'w') as file_output:
	json.dump(output, file_output, indent=4)
