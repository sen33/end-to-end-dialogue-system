from argparse import ArgumentParser

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from tqdm import tqdm
import os

def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples['source']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

parser = ArgumentParser(description='Generate result in certain format.')
parser.add_argument("--model_checkpoint", type=str, help="Checkpoint name or directory", default = '')
parser.add_argument("--output_dir", type=str, help="Output directory", default = 'result/bart-0-5075')
parser.add_argument("--max_input_length", type=int, help="Max input length", default = 512)
parser.add_argument("--max_target_length", type=int, help="Max target length", default = 128)
parser.add_argument("--type", type=str, help="Data type, with-kb, without-kb", default = '')

args = parser.parse_args()

raw_datasets = load_dataset('json', data_files={
    'test': 'data/' + args.type + '/combined-test.json',
    }, field = 'data')
metric = load_metric("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

if args.model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "answer camrest question: "
else:
    prefix = ""

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

os.mkdir(args.output_dir)
file_output = open(f'{args.output_dir}/pred-test.txt', 'w')
for i in tqdm(range(len(tokenized_datasets['test']))):
  data_input = tokenizer(tokenized_datasets['test'][i]['source'], return_tensors='pt')
  result = model.generate(input_ids=data_input['input_ids'], attention_mask=data_input['attention_mask'], min_length=1, max_length=512)
  with tokenizer.as_target_tokenizer():
    generated_sentence = tokenizer.decode(result[0], skip_special_tokens=True)
    file_output.write(generated_sentence)
    file_output.write('\n')
