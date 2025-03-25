import whisperx
from whisperx.utils import WriteSRT

LANGUAGE = "ja"  # Make sure to specify the language if you know it

# Must be UTF-8 encoding
TEXT = "FANTASIA"

# Include extension i.e., .wav, .mp3
AUDIO = "nbb_1.wav"

DEVICE = "cpu"

# Load reference text file, preserving order
with open(f"text/{TEXT}.txt", "r", encoding="utf-8") as f:
    reference_lines = [line.strip() for line in f if line.strip()]

# Combine reference lines into a single transcript
reference_transcript = " ".join(reference_lines)

# Load alignment model
alignment_model, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)

# Align with reference transcript
aligned_result = whisperx.align_with_transcript(
    transcript=reference_transcript,
    model=alignment_model,
    metadata=metadata,
    audio=f"audio/{AUDIO}",
    device=DEVICE
)

# Make sure language is set
if "language" not in aligned_result:
    aligned_result["language"] = LANGUAGE

srt_writer = WriteSRT()

# Write the SRT file
with open(f"srt/{TEXT}.srt", "w", encoding="utf-8") as file_obj:
    srt_writer.write_result(
        result=aligned_result,
        file=file_obj,
        options={
            "word_timestamps": True,
            "max_line_width": 42,
            "max_line_count": 1,
            "highlight_words": False,
            "preserve_segments": True
        }
    )

print(f"\n✅ SRT file saved: srt/{TEXT}.srt")