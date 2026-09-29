min_pixels=$((512*512))
sample_num=5000


python to_fixretina.py \
    --in_path data_v0.8_visual_toolbox_v2.parquet \
    --out_path dpys_chart_valina_${min_pixels}toNone.parquet \
    --min_pixels $min_pixels \
    --num_workers 10 \
    --compression_ratio 1 \
    --sample_num $sample_num