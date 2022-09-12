import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

model = deeplabv3_resnet50(pretrained=False, progress=True, aux_loss=False)
model.classifier = DeepLabHead(in_channels = 2048, num_classes = 2)
model_state_dict = torch.load("../model/ckpt/ckpt_bb/best.pth")["model"]
model.load_state_dict(model_state_dict, strict=True)
model.eval()

example = torch.rand(1,3,298,224)

traced_script_module = torch.jit.trace(model, example, strict=False)

traced_script_module.save("traced_model.pt")