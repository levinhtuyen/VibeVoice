from transformers import AutoConfig
import traceback

c = AutoConfig.from_pretrained('microsoft/VibeVoice-ASR')
print('type', type(c))
print('torch_dtype', getattr(c, 'torch_dtype', None))
try:
    s = c.to_json_string()
    print('to_json_string OK')
    print(s[:800])
except Exception as e:
    print('to_json failed', type(e), e)
    traceback.print_exc()
