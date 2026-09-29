"""
Curriculum learning utilities for dynamic B_{pix} (pixel budget) scheduling.

The pixel budget per image is linearly decayed from `b_pix_start` to `b_pix_end`
over the course of training.

Usage in config (hydra / OmegaConf):
    curriculum:
        b_pix_start: 200704   # 16 * 28 * 16 * 28
        b_pix_end:   50176    # 8 * 28 * 8 * 28
"""

import math
from typing import Tuple
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def bt2px(bt):
    # token budget to pixel budget
    return bt * IMAGE_FACTOR**2

# def compute_curriculum_pixel_budget(
#     b_pix_start: int,
#     b_pix_end: int,
#     global_training_steps: int,
#     total_training_steps: int,
# ) -> int:
#     if total_training_steps <= 1:
#         # If there is only one step, return the starting budget directly.
#         return b_pix_start

#     # 1. Calculate the starting and ending token budgets
#     b_token_start = b_pix_start // IMAGE_FACTOR**2
#     b_token_end = b_pix_end // IMAGE_FACTOR**2

#     # 2. Compute the progress ratio (linear decay between 0 and 1)
#     #    Ensure progress is clamped between 0 and 1 to avoid overflow.
#     progress = min(max((global_training_steps - 1) / (total_training_steps - 1), 0.0), 1.0)

#     # 3. Compute the current token budget using linear interpolation
#     raw_budget = b_token_start + (b_token_end - b_token_start) * progress

#     # Ensure the raw budget is not less than the ending token budget
#     raw_budget = max(raw_budget, b_token_end)

#     # 4. Convert the token budget back to pixel budget, ensuring snapping
#     factor_sq = IMAGE_FACTOR * IMAGE_FACTOR
#     snapped = int(round(raw_budget)) * factor_sq

#     # Constrain the snapped pixel budget within the global limits
#     snapped = max(snapped, MIN_PIXELS)
#     snapped = min(snapped, MAX_PIXELS)

#     # B_{t+1} = B_t - \alpha (passsrate - 0.5)

#     return snapped





# class CurriculumController:
#     def __init__(
#         self,
#         b_start: int = 256,
#         b_min: int = 64,
#         b_max: int = 512,
#         alpha: float = 100.0,
#     ):
#         self.b_tkn = b_start # current token budget
#         self.b_pix = bt2px(self.b_tkn) # current pixel budget
#         self.b_min = b_min # min token budget
#         self.b_max = b_max # max token budget
#         self.alpha = alpha # adjust the speed,

#     def step(self, passrate) -> Tuple[int, int]:
#         """
#         passrate: passrate within all **generated**(not batches for training) 
#         batch within one DAPO update
#         called after each DAPO Update and new passrate calculation.
#         return: new token budget and pixel budget
#         """
#         passrate = float(passrate)
#         if self.alpha:
#             self.b_tkn -= self.alpha * (passrate - 0.5)
#             self.b_tkn = int(round(max(self.b_min, min(self.b_max, self.b_tkn)))) # clamp
#             self.b_pix = bt2px(self.b_tkn) # update pixel budget
#             return self.b_tkn, self.b_pix

class CurriculumController: # linear
    def __init__(
        self,
        b_start: int = 256,
        b_end: int = 169,
        warmup_steps: int = 40  # number of steps over which to decay
    ):
        self.b_start = b_start
        self.b_end = b_end
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.b_tkn = b_start
        self.b_pix = bt2px(self.b_tkn)

    def step(self, passrate=None) -> Tuple[int, int]:
        """
        Called after each DAPO update.
        passrate is accepted but ignored — budget follows pure linear decay.
        Linear decay from b_start to b_end over total_steps, then plateau at b_end.
        """
        self.current_step += 1 
        progress = min(self.current_step / self.warmup_steps, 1.0) # 0 -> 1
        raw = self.b_start + (self.b_end - self.b_start) * progress
        self.b_tkn = int(round(max(raw, self.b_end)))
        self.b_pix = bt2px(self.b_tkn)
        return self.b_tkn, self.b_pix


# class CurriculumController:
#     def __init__(
#         self, 
#         r_start: float = 0.3,
#         r_min: float = 0.2,
#         r_max: float = 0.5,
#         beta: float = 0.05,
#     ):
#         self.r_current = r_start
#         self.r_min = r_min
#         self.r_max
#         self.beta = beta

#     def step(self, passrate):
#         passrate = float(passrate)
#         if self.beta:
#             self.r_current -= self.beta * (passrate - 0.5)
#             self.r_current = max(self.r_min, min(self.r_max, self.r_current))
#             return self.r_current
            
class RatioCurriculumController:
    """
    Budget expressed as a ratio (0 < r < 1) of original image size.
    B_{t+1} = B_t - alpha * (passrate - 0.5)
    passrate > 0.5 → too easy → shrink ratio (harder)
    passrate < 0.5 → too hard → grow ratio (easier)
    """
    def __init__(
        self,
        r_start: float = 0.2,
        r_min: float = 0.2,
        r_max: float = 0.2,
        alpha: float = 0.02,  
    ):
        self.ratio = r_start
        self.r_min = r_min
        self.r_max = r_max
        self.alpha = alpha
        print(f"[DEBUG] Init Ratio Controler with {r_start=}, {r_min=}, {r_max=}, {alpha=}")

    def step(self, passrate) -> Tuple[None, float]:
        """
        Returns (token_budget=None, pixel_budget=ratio).
        ratio is in (0, 1), triggering the dataset's ratio mode.
        """
        passrate = float(passrate)
        if passrate > 0.5: 
            self.ratio -= self.alpha
        else:
            self.ratio += self.alpha
        self.ratio = round(max(self.r_min, min(self.r_max, self.ratio)), 4)
        return None, self.ratio
