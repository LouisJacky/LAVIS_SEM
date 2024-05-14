import torch
import os
from collections import OrderedDict

project_id = "your_project_id" # eg: 20240509214
path_to_LAVIS_SEM ="/path_to_your/LAVIS_SEM"

# 加载模型参数
params_pretrained = torch.load('https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth')
params_finedtuned = torch.load(f'{path_to_LAVIS_SEM}/lavis/output/SEM/finetuned/{project_id}/checkpoint_best.pth')


# 更新模型参数
# 创建一个新的OrderedDict
new_model_weights = OrderedDict()
new_model_weights['model'] = OrderedDict()

for name, param in params_pretrained['model'].items():
    new_model_weights['model'][name] = param

for name, param in params_finedtuned['model'].items():
    new_model_weights['model'][name] = param

# 保存合并后的模型参数
torch.save(new_model_weights, f'{path_to_LAVIS_SEM}/SEM/Checkpoints/merged_{project_id}.pth')