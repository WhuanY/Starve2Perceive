## Training Dataset Description
To truly validate if compression visual reasoning environment indeed fosters agent's capability to reason and solve complex VQA tasks in an activate manner, we investigated and sampled diverse RL datasource tailored for VQA tasks. 


### Sampled Data Source
- [ViRL5k](https://arxiv.org/abs/2511.23031): This work sampled 200k-scale dataset by collecting samples from Visual-Cot, GQA, TextVQA, Visual7W, InfographicsVQA, and GRIT, covering general understanding, visual grounding, reasoning, and OCR tasks. This work filtered high-quality samples by policy generation, answer verification and reasoning-centric filtering, finally retain around 8k of them. We sampled **2722** of them for RL training. 
- [VisualProbe](https://mini-o3.github.io/): This work utilized comprises 4,000 visual question–answer pairs for training. VisualProbe is characterized by: 1) Small targets 2) Numerous distractor objects 3) High-resolution images. This work fits the motivation of our works well. We utilize 2k of their SFT trajectories and re-writed them for our sft-training, and their proposed training set for RL training.
- [DeepEyes](https://arxiv.org/abs/2505.14362): This work utilized VQAs composed from from V*(22k), ArxivQA(13k) and ThinkLite-VL(11k). We utilize the ArxivQA subset by sampling 5k of them for RL training.
- [TreeVGR](https://arxiv.org/abs/2507.07999) This work proposed TreeBench, a benchmark for evaluating "thinking with images" capacities with tracable visual evidence. 

### Adapt Existing Dataset to Compression Setting.
Directly using the original image format and image resolution may not be suitable for compression visual reasoning environment. We adapt the existing dataset to compression setting by:
1. Applying a minimum pixel threshold to filter out images with too few pixels. The minimum threshold Our default setting is $512 \times 512$. 
2. We keep the original image to data column `seed_image`. Then we prepare the `overview_image` by applying a fixed-size crop($16 \times 16 \times 28 \times 28$) to any original image higher than the threshold. We store the `scale_x = overview.w / original.w` and `scale_y = overview.h / original.h` to data column `scale_x` and `scale_y`. This column stores the mapping relationship between the `original_image` and the `overview_image`. Enabling the visual agent to reason about the `original_image` by zooming in on the `overview_image` with the scale information.

## Appendix
### Data Composition
From ViRL5k: 2722 records
From DeepEyes: 5000 records of chart RL Training Dataset 
From VisualProbe: 5729 records of visual search reasoning 
From TreeVGR: 5570 records visual search.
From PixelReasoner: 4539 records visual search and doc understanding
### Prompt
1. System Prompt Setting:
```
"You are a helpful assistant.

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
```
2. User Prompt Setting
```
<image>\nThink first, call **focus** if needed, then answer if you are confident. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>  You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Here is the question:\n
```
