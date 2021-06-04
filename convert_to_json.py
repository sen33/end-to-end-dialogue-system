import json
from argparse import ArgumentParser

parser = ArgumentParser(description='Generate json data.')
parser.add_argument("--type", type=str, help="empty, with-kb, or without-kb", default='')

args = parser.parse_args()

file_path = 'data/' + args.type
dataset_types = ['test', 'dev', 'train']
for dataset_type in dataset_types:
	file_src = open(f'{file_path}/src-{dataset_type}.txt', 'r')
	file_tgt = open(f'{file_path}/tgt-{dataset_type}.txt', 'r')
	src = file_src.read()
	src = src.split('\n')
	tgt = file_tgt.read()
	tgt = tgt.split('\n')

	output = [{} for i in range(len(src) - 1)]
	for i in range(len(src)):
		if i != len(src) - 1:
			output[i]['source'] = src[i]
			output[i]['target'] = tgt[i]

	with open(f'{file_path}/combined-{dataset_type}.json', 'w') as outfile:
		json.dump({'data': output}, outfile, indent=2)
