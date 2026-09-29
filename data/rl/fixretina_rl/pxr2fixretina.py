import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MAX_VIEW_PIXELS = (28 * 16) ** 2  # 200704

NEW_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"focus","description":"Request a high-resolution local region of the first image and zoom in","parameters":{"type":"object","properties":{"bboxes":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to crop, as [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES of the first image."},"description":"A list of bounding boxes to zoom in on. You can request 1-3 bboxes at a turn."}},"required":["bboxes"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "focus", "arguments": {"bboxes": [[10, 20, 100, 200]]}}  
</tool_call>
"""


def change_row(row_dict):
    # 1. Change system prompt
    prompt = list(row_dict['prompt'])
    prompt[0] = dict(prompt[0])
    prompt[0]['content'] = NEW_SYSTEM_PROMPT
    row_dict['prompt'] = np.array(prompt, dtype=object)

    # 2. Add ability column
    row_dict['ability'] = 'vqa'

    # 3. Add ground_truth_bboxes, path, question to reward_model
    rm = dict(row_dict['reward_model'])
    rm['ground_truth_bboxes'] = None
    rm['path'] = None
    rm['question'] = row_dict['extra_info'].get('question', '')
    row_dict['reward_model'] = rm 
    new_img = np.array([{'bytes': row_dict['images'][0]['bytes'], 'path': None}], dtype=object)
    row_dict['images'] = new_img
    # images = list(row_dict['images'])
    # for j in range(len(images)):
    #     img = dict(images[j])
    #     img['path'] = None
    #     images[j] = img
    # row_dict['images'] = np.array(images, dtype=object)

    # 4. Rebuild extra_info: add pixel_budget_per_image, normalize options
    old_ei = row_dict['extra_info']
    options = old_ei.get('options', None)
    if isinstance(options, (list, np.ndarray)) and len(options) == 0:
        options = None
    elif isinstance(options, np.ndarray):
        options = options.tolist()
    
    answer = old_ei.get('answer')
    if isinstance(answer, np.ndarray):
        answer = answer.tolist()

    scale = old_ei.get('seed_img').get('scale')
    if isinstance(scale, np.ndarray):
        scale = scale.tolist()

    row_dict['extra_info'] = {
        'answer': answer,
        'index': old_ei.get('index'),
        'options': options,
        'question': old_ei.get('question'),
        'seed_img': {
            'bytes': old_ei.get('seed_img').get('bytes'),
            'scale': scale,
        },
        'split': old_ei.get('split', 'train'),
        'pixel_budget_per_image': MAX_VIEW_PIXELS,
    }

    return row_dict


if __name__ == "__main__":
    for i in range(5):
        path = f"/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/fixretina_rl_pxr_hires_{i}.parquet"
        print(f"Processing {path} ...")
        df = pd.read_parquet(path)
        rows = df.to_dict(orient='records')
        new_rows = [change_row(r) for r in rows]
        out_path = f"/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/pxr_res1024toNone_fixretina_ap_part{i}.parquet"
        table = pa.Table.from_pylist(new_rows)
        pq.write_table(table, out_path)
        print(f"  Saved {len(new_rows)} rows -> {out_path}")
    print("Done.")
