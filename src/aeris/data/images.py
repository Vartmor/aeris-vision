"""RGB tensor preprocessing shared by both model stages."""

import numpy as np
import torch

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
    return image.sub_(_MEAN).div_(_STD)
