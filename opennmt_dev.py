from argparse import ArgumentParser

parser = ArgumentParser(description='Generate json data.')
parser.add_argument("--type", type=str, help="empty, with-kb, or without-kb", default='')

args = parser.parse_args()

dataset_path = 'data/' + args.type

file_path = 'data/CamRest/'
dataset_types = ['test', 'dev', 'train']
for dataset_type in dataset_types:
	file_dial = open(f'{file_path}{dataset_type}.txt', 'r')
	dialogues = file_dial.read()
	dialogues = dialogues.split('\n\n')

	file_dataset_src = open(f'{dataset_path}/src-{dataset_type}.txt', 'w')
	file_dataset_tgt = open(f'{dataset_path}/tgt-{dataset_type}.txt', 'w')

	print(f'Banyaknya dialog: {dataset_type}', len(dialogues) - 1)

	for dialogue in dialogues:
		utterance = []
		kb = ''
		# conv_sys = []
		line = dialogue.split('\n')
		current_kb = ''
		for conv in line:
			striped_conv = ' '.join(conv.split(' ')[1::])
			if '\t' in conv:
				if striped_conv == '':
					continue
				usr, sys = striped_conv.split('\t')
				if("<SILENCE>" not in usr):
					utterance.append(usr)
				if("i'm on it" not in sys and "api_call" not in sys and "ok let me look into some options for you" not in sys):
					utterance.append(sys)
			elif args.type == 'with-kb' and striped_conv:
				# kb += '<dta> '
				# kb += striped_conv + ' '
				kb_instance = striped_conv.split(' ')
				if current_kb == kb_instance[0]:
					kb += kb_instance[2] + ' '
				else:
					current_kb = kb_instance[0]
					kb += '<dta> '
					kb += kb_instance[0] + ' '
					kb += kb_instance[2] + ' '

		temp = ''
		for i in range(len(utterance)):
			if (i % 2 == 0):
				# if args.type == 'with-kb':
				# 	file_dataset_src.write('<usr> ' + utterance[i] + ' ' + kb + temp)
				if i == 0:
					temp += '<usr> ' + utterance[i]
				else:
					temp += ' <usr> ' + utterance[i]
				if args.type != 'with-kb':
					file_dataset_src.write(temp)
				else:
					file_dataset_src.write(temp + ' ' + kb)
				file_dataset_src.write('\n')
			else:
				temp += ' <sys> ' + utterance[i]
				file_dataset_tgt.write(utterance[i])
				file_dataset_tgt.write('\n')

