import torch
from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

orig = VibeVoiceASRConfig.from_pretrained

def wrapped(*args, **kwargs):
    print('wrapped from_pretrained args', args, 'kwargs', kwargs)
    c = orig(*args, **kwargs)
    if isinstance(c, tuple):
        cfg = c[0]
        print('returned tuple, config dtype attr', getattr(cfg, 'dtype', None), 'torch_dtype', getattr(cfg, 'torch_dtype', None))
    else:
        print('returned config', getattr(c, 'dtype', None), getattr(c, 'torch_dtype', None))
    return c

VibeVoiceASRConfig.from_pretrained = wrapped

try:
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        'microsoft/VibeVoice-ASR',
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation='sdpa'
    )
    print('model loaded', type(model))
except Exception as e:
    import traceback
    print('error', type(e), e)
    traceback.print_exc()
