import torch
from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig

c = VibeVoiceASRConfig(torch_dtype=torch.float32)
print('c', c)
print('json ok', c.to_json_string()[:400])
