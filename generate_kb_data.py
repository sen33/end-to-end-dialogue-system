import json

kb_file = open('data/CamRest/KB.json')

kb = json.load(kb_file)

kb_data = ''
for i in range(len(kb)):
	kb_data += '<dta> '
	for key in kb[i]:
		if(key == "postcode"):
			kb[i][key] = kb[i][key].replace(".","").replace(",","").replace(" ","").lower()
		else: 
			kb[i][key] = kb[i][key].replace(" ","_").lower()
	kb_data += kb[i]['address'] + ' '
	kb_data += kb[i]['area'] + ' '
	if 'food' in kb[i]:
		kb_data += kb[i]['food'] + ' '
	else:
		kb_data += 'international' + ' '
	kb_data += kb[i]['location'] + ' '
	if 'phone' in kb[i]:
		kb_data += kb[i]['phone'] + ' '
	else:
		kb_data += '01223_000000' + ' '
	kb_data += kb[i]['pricerange'] + ' '
	kb_data += kb[i]['postcode'] + ' '
	kb_data += kb[i]['type'] + ' '
	kb_data += kb[i]['id'] + ' '
	kb_data += kb[i]['name'] + ' '

file_path = 'data/with-kb/'
dataset_types = ['test', 'dev', 'train']

for dataset_type in dataset_types:
	file_src = open(f'{file_path}src-{dataset_type}.txt', 'r')
	utterances = file_src.read()
	utterances = utterances.split('\n')
	file_src.close()

	file_src = open(f'{file_path}src-{dataset_type}.txt', 'w')

	for utterance in utterances:
		file_src.write(kb_data + utterance)
		file_src.write('\n')
