from argparse import ArgumentParser
from tqdm import tqdm

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

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
parser.add_argument("--model_checkpoint", type=str, help="Checkpoint name or directory", default = 'facebook/bart-base')
parser.add_argument("--batch_size", type=int, help="Batch size", default = 8)
parser.add_argument("--learning_rate", type=float, help="Learing rate", default = 1e-5)
parser.add_argument("--epoch", type=int, help="Epoch", default = 10)
parser.add_argument("--output_dir", type=str, help="Output directory", default = 'run/bart/0')
parser.add_argument("--type", type=str, help="Data type", default = '')
parser.add_argument("--max_input_length", type=int, help="Max input length", default = 512)
parser.add_argument("--max_target_length", type=int, help="Max target length", default = 128)

args = parser.parse_args()

raw_datasets = load_dataset('json', data_files={
    'train': 'data/' + args.type + '/combined-train.json',
    'test': 'data/' + args.type + '/combined-test.json',
    'validation': 'data/' + args.type + '/combined-dev.json',
    }, field = 'data')
metric = load_metric("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

if args.type == 'with-kb':
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<usr>', '<sys>', '<dta>']}
else:
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<usr>', '<sys>']}
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
model.resize_token_embeddings(len(tokenizer))

if args.model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "answer camrest question: "
else:
    prefix = ""

print('PREFIX: ' + prefix)

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

trainingArgs = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy = "epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=args.epoch,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model='bleu',
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    trainingArgs,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
