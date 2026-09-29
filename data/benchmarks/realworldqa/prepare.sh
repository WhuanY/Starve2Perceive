cd $(dirname $0)

wget https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv

python tsv2json.py

rm *.tsv

