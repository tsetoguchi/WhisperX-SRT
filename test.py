import whisperx
from whisperx.utils import WriteSRT

# Configuration
AUDIO = "Anyone.mp3"
LANGUAGE = "en"
MODEL = "large-v2"
DEVICE = "cpu"

# Load model
model = whisperx.load_model(
    MODEL,
    device=DEVICE,
    compute_type="float32"
)

# Transcribe with VAD filtering
result = model.transcribe(
    f"audio/{AUDIO}",
    language=LANGUAGE,
)

# Load alignment model
alignment_model, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)

# Align
aligned_result = whisperx.align(
    result["segments"],
    alignment_model,
    metadata,
    f"audio/{AUDIO}",
    DEVICE
)

# Write SRT
writer = WriteSRT("srt/")
with open(f"{AUDIO.rsplit('.', 1)[0]}.srt", "w", encoding="utf-8") as file_obj:
    writer.write_result(
        result=aligned_result,
        file=file_obj,
        options={
            "word_timestamps": True,
            "max_line_width": 42,
            "max_line_count": 2,
            "highlight_words": False
        }

    )

print(f"✅ SRT file saved: {AUDIO.rsplit('.', 1)[0]}.srt")