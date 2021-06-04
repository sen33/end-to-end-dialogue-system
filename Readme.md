`
pip install -r requirements.txt
`

Buat file pasangan pertanyaan jawaban dari train, test, dan dev
`
python opennmt_dev.py
`

Tambah file KE ke train dataset
`
python opennmt.py
`

Mengubah bentuk dataset ke json
`
python convert_to_json.py
`

### OpenNMT
`
./script.sh <nomor konfigurasi>
`

### BART / T5
`
python train.py --model_checkpoint <jenis_model> --batch_size <batch_size> --learning_rate <lr> --epoch <max epoch> --output_dir <directory output>
`