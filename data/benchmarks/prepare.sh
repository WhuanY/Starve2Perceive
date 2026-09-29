cd $(dirname $0)

# mmerealworldlite
# aria2c -x 16 -s 16 -o data.zip https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite/resolve/main/data.zip
# unzip data.zip
# mv data/ mmerealworldlite/
# rm data.zip

# vstar_bench
# hf download craigwu/vstar_bench --repo-type dataset --local-dir vstar_bench

# hrbench
# mkdir -p hrbench
# cd hrbench
# aria2c -x 16 -s 16  https://hf-mirror.com/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv
# aria2c -x 16 -s 16  https://hf-mirror.com/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv
# cd ..



