import duckdb
import os

# 输入和输出文件路径

input_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/virl8k_crNone_res_262144toNone_fixretina_ap.parquet"
output_file = "/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/virl8k_crNone_res_262144toNone_fixretina_ap_100.parquet"

# 检查输入文件是否存在
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

# 使用 DuckDB 从 Parquet 文件随机筛选 10 条记录
query = f"""
COPY (
    SELECT *
    FROM read_parquet('{input_file}')
    USING SAMPLE 100
) TO '{output_file}' (FORMAT PARQUET);
"""

# 执行查询
duckdb.query(query)

print(f"100 random rows have been saved to: {output_file}")