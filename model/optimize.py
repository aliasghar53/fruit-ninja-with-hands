import tensorrt
import torch_tensorrt
from pathlib import Path
import sys
import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, ConcatDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model.dataset import EgoHands, FreiHands
from model.multi_task_scheduler import BatchSchedulerSampler


# build and load model weights
model = deeplabv3_resnet50(progress=True, aux_loss=False)
model.classifier = DeepLabHead(in_channels = 2048, num_classes = 2)
model_state_dict = torch.load("../model/ckpt/ckpt_bb/best.pth")["model"]
model.load_state_dict(model_state_dict, strict=True)
model.eval()

sample_input = torch.empty([1,3,298,224])
traced_model = torch.jit.trace(model, sample_input, strict=False)

traced_model.to("cuda:0")

# ego_hands = EgoHands(mode="eval", size=(298,224))
frei_hands = FreiHands(mode="eval", size=(298,224))

# dataset = ConcatDataset([ego_hands, frei_hands])

data_loader = DataLoader(
                        frei_hands,
                        batch_size = 4,
                        # sampler=BatchSchedulerSampler(dataset, 1), 
                        shuffle=True, 
                        num_workers = 1,
                        pin_memory=True                              
                    )

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    data_loader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda:0"),
)

trt_model = torch_tensorrt.compile(traced_model, inputs=[torch_tensorrt.Input((1, 3, 298, 224))],
                                    enabled_precisions={torch.float, torch.half, torch.int8},
                                    calibrator=calibrator,
                                    device={
                                         "device_type": torch_tensorrt.DeviceType.GPU,
                                         "gpu_id": 0,
                                         "dla_core": 0,
                                         "allow_gpu_fallback": False,
                                         "disable_tf32": False
                                     })

trt_model.save("optimized_best.ts")
