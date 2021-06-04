# !/bin/bash

echo "Running Script with id = $1"

onmt_train -config config/camrest_config_$1.yaml

onmt_translate -model run/$1/model_step_50000.pt -src data/src-test.txt -output result/$1/pred-test.txt

python3 opennmt_generate_result.py --model_id $1

python3 scorer_CAMREST.py
