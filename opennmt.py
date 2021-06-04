file_path = 'data/CamRest/'
file_dial = open(file_path + 'gen-babi7-nk201-nd156-rs0.txt', 'r')
dialogues = file_dial.read()
dialogues = dialogues.split('\n\n')

file_dataset_src = open('data/src-train.txt', 'a')
file_dataset_tgt = open('data/tgt-train.txt', 'a')

print('Banyaknya dialog: ', len(dialogues))

for dialogue in dialogues:
	utterance = []
	line = dialogue.split('\n')
	for conv in line:
		striped_conv = ' '.join(conv.split(' ')[1::])
		if striped_conv == '':
			continue
		usr, sys = striped_conv.split('\t')
		if("<SILENCE>" not in usr):
				utterance.append(usr)
		if("i'm on it" not in sys and "api_call" not in sys and "ok let me look into some options for you" not in sys):
				utterance.append(sys)

	temp = ''
	for i in range(len(utterance)):
		if (i % 2 == 0):
			if i == 0:
				temp += '<usr> ' + utterance[i]
			else:
				temp += ' <usr> ' + utterance[i]
			file_dataset_src.write(temp)
			file_dataset_src.write('\n')
		else:
			temp += ' <sys> ' + utterance[i]
			file_dataset_tgt.write(utterance[i])
			file_dataset_tgt.write('\n')
