from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
import traceback

try:
    proc = VibeVoiceASRProcessor.from_pretrained('microsoft/VibeVoice-ASR')
    print('processor ok', proc)
except Exception as e:
    print('proc err', repr(e))
    traceback.print_exc()

try:
    model = VibeVoiceASRForConditionalGeneration.from_pretrained('microsoft/VibeVoice-ASR', dtype='float32', trust_remote_code=True)
    print('model ok', model)
except Exception as e:
    print('model err', type(e), e)
    traceback.print_exc()
