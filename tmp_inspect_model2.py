import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
import traceback

try:
    proc = VibeVoiceASRProcessor.from_pretrained('microsoft/VibeVoice-ASR')
    print('processor ok')
except Exception as e:
    print('proc err', repr(e))
    traceback.print_exc()

for dtype in ['float32', 'bfloat16']:
    try:
        print('loading dtype', dtype)
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            'microsoft/VibeVoice-ASR',
            dtype=torch.bfloat16 if dtype=='bfloat16' else torch.float32,
            trust_remote_code=True
        )
        print('model ok', dtype)
        cfg = model.config.to_dict()
        print('cfg torch_dtype', cfg.get('torch_dtype'))
    except Exception as e:
        print('model err', dtype, type(e), e)
        traceback.print_exc()
