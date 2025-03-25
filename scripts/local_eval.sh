echo "configuration file: $1; exam dataset files: $2; output path: $3"

python imagination_aimo2/local_eval.py $1 --seed 123 --exam-dataset-files $2 --output-path $3
