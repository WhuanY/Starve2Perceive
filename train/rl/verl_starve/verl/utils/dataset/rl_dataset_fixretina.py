from verl.utils.dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from multiprocessing import Value
from PIL import Image
import io

class RLHFDatasetFixRetina(RLHFDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._current_pixel_budget_shared = Value('d', -1.0)  # -1.0 表示未设置
        self.current_pixel_budget = None  # 保持兼容性，但实际从 shared value 读取

    @property
    def current_pixel_budget(self):
        """从共享内存读取 current_pixel_budget"""
        val = self._current_pixel_budget_shared.value
        return None if val == -1.0 else val
    
    @current_pixel_budget.setter
    def current_pixel_budget(self, value):
        """设置共享内存中的 current_pixel_budget"""
        if value is None:
            self._current_pixel_budget_shared.value = -1.0
        else:
            self._current_pixel_budget_shared.value = float(value)

    def __getitem__(self, item):
        budget = self.current_pixel_budget
        print(f"[DEBUG] RLHFDatasetFixRetina.__getitem__, {budget=}")
        if budget is None:
            return super().__getitem__(item)
        else:
            row_dict: dict = self.dataframe[item]
            messages = self._build_messages(row_dict)
            model_inputs = {}

            if self.processor is not None:

                from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video, constrain_image_size

                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                multi_modal_data = {}
                origin_multi_modal_data = {}

                images = None
                seed_img_dict_copy = None
                if self.image_key in row_dict:
                    seed_img_dict = row_dict.get("extra_info").get("seed_img")
                    seed_img_dict_copy = dict(seed_img_dict)

                    assert 'bytes' in seed_img_dict.keys(), "bytes not found in seed_img_dict"
                    seed_img_bytes = seed_img_dict.get('bytes')
                    seed_img_pil = Image.open(io.BytesIO(seed_img_bytes)) # 获得原图
                    max_pixels = 16 * 16 * 28 * 28 # initial value
                    if 0 < budget < 1:
                        seed_img_pixels = seed_img_pil.width * seed_img_pil.height
                        max_pixels = int(seed_img_pixels * budget * budget) #budget 就是ratio 
                        # we clampped it to max_pixels \in [13 * 13 * 28 * 28, 1024 * 1024]
                        max_pixels = max(13 * 13 * 28 * 28, min(max_pixels, 1024 * 1024)) # 在clamp下，我们仍然限制budget在169 到1024 * 1024 / 28 / 28之间
                    else:
                        max_pixels = int(budget)
                    overview_img, scale = constrain_image_size(
                        seed_img_pil,
                        max_pixels = max_pixels
                    )
                    overview_img = overview_img.convert("RGB")
                    seed_img_dict_copy['scale'] = scale # update scale in place
                    # overview_img to bytes 
                    buf = io.BytesIO()
                    overview_img.save(buf, format="JPEG")
                    overview_img_bytes = buf.getvalue()
                    # hack the self.image_key field
                    row_dict[self.image_key] = [{"bytes": overview_img_bytes, "path": None}]
                    # process the images
                    origin_images = [process_raw_image(image) for image in row_dict.get(self.image_key)]
                    images = [process_image(image) for image in row_dict.pop(self.image_key)]
                    
                    multi_modal_data["image"] = images
                    origin_multi_modal_data["image"] = origin_images

                videos = None
                if self.video_key in row_dict:
                    videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                    multi_modal_data["video"] = [video.numpy() for video in videos]

                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")

                if "second_per_grid_ts" in model_inputs:
                    model_inputs.pop("second_per_grid_ts")

                # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
                row_dict['origin_multi_modal_data'] = origin_multi_modal_data
                row_dict["multi_modal_data"] = multi_modal_data
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            else:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index

                position_ids = [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ]  # (1, 3, seq_len)

            else:
                position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]

            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif self.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

            row_dict["raw_prompt_ids"] = raw_prompt_ids
            # encode prompts without chat template
            if self.return_raw_chat:
                row_dict["raw_prompt"] = messages

            # add index for each prompt
            index = row_dict.get("extra_info", {}).get("index", 0)
            row_dict["index"] = index
            
            # fixretina: need a high-res image for tool use
            if seed_img_dict_copy is not None:
                row_dict['seed_img'] = seed_img_dict_copy
            else:
                row_dict['seed_img'] = row_dict.get('extra_info', {}).get('seed_img', None)
            row_dict['pixel_budget_per_image'] = max_pixels
            print(f"[DEBUG] {row_dict['pixel_budget_per_image']=} ")
            return row_dict
