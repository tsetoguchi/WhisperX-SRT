import whisperx
from whisperx.utils import WriteSRT


LANGUAGE = "en"
TEXT = "Anyone"
AUDIO = "Anyone.mp3"
DEVICE = "cpu"
MODEL_SIZE = "medium"

# Load transcription model
model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type="float32")

# Transcribe audio
result = model.transcribe(f"audio/{AUDIO}", language=LANGUAGE)

result["language"] = LANGUAGE

# Load reference text file
with open(f"text/{TEXT}.txt", "r", encoding="utf-8") as f:
    reference_lines = [line.strip() for line in f if line.strip()]

# Extract transcribed segments
transcribed_lines = [segment['text'] for segment in result['segments']]

# Load alignment model
alignment_model, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)

for segment in result["segments"]:
    print(segment)


# Write SRT file
srt_writer = WriteSRT("srt/")
with open(f"srt/{TEXT}.srt", "w", encoding="utf-8") as file_obj:
    srt_writer.write_result(
        result=result,
        file=file_obj,
        options={
            "word_timestamps": True,
            "max_line_width": 1000,
            "max_line_count": 1,
            "highlight_words": False,
            "preserve_segments": True,
            "language": LANGUAGE
        }
    )

print(f"\n✅ SRT file saved: srt/{TEXT}.srt")