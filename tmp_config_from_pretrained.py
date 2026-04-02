from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
import traceback

for kw in [dict(), dict(torch_dtype='float32'), dict(torch_dtype='bfloat16'), dict(dtype='float32'), dict(dtype='bfloat16')]:
    try:
        print('trying', kw)
        c = VibeVoiceASRConfig.from_pretrained('microsoft/VibeVoice-ASR', **kw)
        print('ok', c)
    except Exception as e:
        print('err', kw, type(e), e)
        traceback.print_exc()
