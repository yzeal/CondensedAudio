#Install dependencies:
# pip install openai-whisper
# pip install ffmpeg-python
# Ensure you have ffmpeg installed on your system

#Usage examples:
# python condense_spoken_audio.py "path/to/your/anime.mp4"
# python condense_spoken_audio.py "path/to/your/anime_folder" --batch
# python condense_spoken_audio.py /path/to/video.mp4 --output /path/to/output --silence 1.0 --model medium --no-skip-intro


import whisper
import subprocess
import os
import shutil
from pathlib import Path
import argparse
import glob

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", str(audio_path)
    ])

def transcribe_audio(audio_path, model_size="medium"):
    model = whisper.load_model(model_size)
    return model.transcribe(str(audio_path))

def get_dialogue_start(segments, min_dialogue_length=1.0):
    """Find the first real dialogue segment, ignoring music and short utterances"""
    # Skip segments that are likely non-dialogue (music, sound effects, etc.)
    non_dialogue_markers = ["♪", "music", "音楽", "(", ")", "[", "]", "*"]
    
    for i, seg in enumerate(segments):
        text = seg["text"].strip().lower()
        
        # Skip if text contains any non-dialogue markers
        if any(marker in text for marker in non_dialogue_markers):
            continue
            
        # Skip very short utterances (likely sound effects)
        if len(text) < 5 or (seg["end"] - seg["start"]) < min_dialogue_length:
            continue
            
        # Check if there are other nearby dialogue segments to confirm this is actual dialogue
        nearby_dialogue = False
        for next_seg in segments[i+1:i+4]:  # Look at next few segments
            if next_seg["start"] - seg["end"] < 5.0:  # Within 5 seconds
                nearby_dialogue = True
                break
                
        if nearby_dialogue:
            return max(seg["start"] - 0.5, 0.0)
            
    return 0.0  # Default to beginning if no dialogue detected

def generate_segments(segments, silence_threshold, max_segment_length=60.0):
    """Generate segments, splitting very long segments and respecting silence threshold"""
    merged = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        
        # Skip segments that are just music notation or very short
        text = seg["text"].strip().lower()
        if text.startswith("♪") or "music" in text or len(text) < 2:
            continue
            
        if not merged:
            merged.append((start, end))
        else:
            gap = start - merged[-1][1]
            if gap >= silence_threshold:
                merged.append((start, end))
            else:
                # Merge with previous segment
                merged[-1] = (merged[-1][0], end)
                
    # Split any excessively long segments
    final_segments = []
    for start, end in merged:
        if end - start > max_segment_length:
            # Split into chunks
            current = start
            while current < end:
                next_point = min(current + max_segment_length, end)
                final_segments.append((current, next_point))
                current = next_point
        else:
            final_segments.append((start, end))
            
    return final_segments

def extract_clips(audio_path, segments, out_dir):
    clip_paths = []
    for i, (start, end) in enumerate(segments):
        clip_path = out_dir / f"clip_{i}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-to", str(end),
            "-c", "copy", str(clip_path)
        ])
        clip_paths.append(clip_path)
    return clip_paths

def concatenate_clips(clip_paths, output_path):
    list_path = output_path.parent / "concat_list.txt"
    with open(list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{path.resolve()}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path), "-c", "copy", str(output_path)
    ])

def convert_to_m4a(wav_path, m4a_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "aac", "-b:a", "128k", str(m4a_path)
    ])

def cleanup_temp_files(audio_path, condensed_path, clips_dir, list_file):
    """Remove all temporary files and directories"""
    if audio_path.exists():
        os.remove(audio_path)
    
    if condensed_path.exists():
        os.remove(condensed_path)
    
    if list_file.exists():
        os.remove(list_file)
        
    if clips_dir.exists() and clips_dir.is_dir():
        shutil.rmtree(clips_dir)

def process_video(video_file, output_dir, silence_threshold, model_size, skip_intro, skip_seconds=None, min_dialogue_length=1.0):
    """Process a single video file"""
    video_path = Path(video_file)
    stem = video_path.stem
    
    # Create temporary working directory
    temp_dir = Path(os.path.dirname(video_file)) / f"temp_{stem}"
    temp_dir.mkdir(exist_ok=True)
    
    audio_path = temp_dir / f"{stem}_full.wav"
    condensed_path = temp_dir / f"{stem}_condensed.wav"
    clips_dir = temp_dir / f"{stem}_clips"
    clips_dir.mkdir(exist_ok=True)
    list_file = temp_dir / "concat_list.txt"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    final_m4a_path = output_dir / f"{stem}_condensed.m4a"

    try:
        print(f"\n🎬 Processing {video_path.name}...")
        print("🔊 Extracting audio...")
        extract_audio(video_path, audio_path)

        print("🧠 Transcribing with Whisper...")
        result = transcribe_audio(audio_path, model_size=model_size)

        if skip_seconds is not None:
            # Explicit time-based skipping
            print(f"⏭️  Skipping first {skip_seconds:.2f} seconds...")
            trimmed_segments = [s for s in result["segments"] if s["end"] > skip_seconds]
        elif skip_intro:
            print("⏭️  Locating first real dialogue...")
            start_offset = get_dialogue_start(result["segments"], min_dialogue_length)
            print(f"👉 Skipping to {start_offset:.2f}s")
            trimmed_segments = [s for s in result["segments"] if s["end"] > start_offset]
            if not trimmed_segments:
                print("⚠️  No dialogue segments found after intro skip. Try --no-skip-intro or check audio content.")
                return False
        else:
            trimmed_segments = result["segments"]

        if not trimmed_segments:
            print("❌ No dialogue segments found at all. Exiting.")
            return False

        print("✂️ Generating dialogue segments...")
        segments = generate_segments(trimmed_segments, silence_threshold=silence_threshold)

        if not segments:
            print("❌ No valid segments generated. Check your silence threshold or try a different model.")
            return False

        print("🎬 Extracting clips...")
        clip_paths = extract_clips(audio_path, segments, clips_dir)

        print("🔗 Concatenating clips...")
        concatenate_clips(clip_paths, condensed_path)

        print("🎧 Compressing to M4A...")
        convert_to_m4a(condensed_path, final_m4a_path)

        print(f"✅ Done! File saved at: {final_m4a_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {str(e)}")
        return False
        
    finally:
        # Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        cleanup_temp_files(audio_path, condensed_path, clips_dir, list_file)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main(video_input, output_dir=None, silence_threshold=1.5, model_size="medium", 
         skip_intro=True, skip_seconds=None, min_dialogue_length=1.0, batch_mode=False):
    
    # Handle batch processing of a directory
    if batch_mode:
        input_dir = Path(video_input)
        if not input_dir.is_dir():
            print(f"❌ Error: {input_dir} is not a directory")
            return
            
        # Set output directory (default: "Condensed Audio" subdirectory)
        if output_dir is None:
            output_dir = input_dir / "Condensed Audio"
        else:
            output_dir = Path(output_dir)
            
        # Get all video files in the directory
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(input_dir.glob(f"*{ext}")))
            
        if not video_files:
            print(f"❌ No video files found in {input_dir}")
            return
            
        print(f"🎬 Found {len(video_files)} video files to process")
        
        success_count = 0
        for video_file in video_files:
            success = process_video(
                video_file, output_dir, silence_threshold, model_size, 
                skip_intro, skip_seconds, min_dialogue_length
            )
            if success:
                success_count += 1
                
        print(f"\n✅ Processed {success_count} of {len(video_files)} videos successfully")
        print(f"📂 Condensed audio files saved to: {output_dir}")
        
    else:
        # Process a single video file
        video_path = Path(video_input)
        if not video_path.is_file():
            print(f"❌ Error: {video_path} is not a file")
            return
            
        # Set output directory
        if output_dir is None:
            output_dir = video_path.parent / "Condensed Audio"
        else:
            output_dir = Path(output_dir)
            
        process_video(
            video_path, output_dir, silence_threshold, model_size,
            skip_intro, skip_seconds, min_dialogue_length
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condense spoken audio from video files.")
    parser.add_argument("input", help="Path to input video file or directory")
    parser.add_argument("--output", help="Path to output directory (default: 'Condensed Audio' subfolder)")
    parser.add_argument("--silence", type=float, default=1.5, help="Silence threshold in seconds (default: 1.5)")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--no-skip-intro", dest="skip_intro", action="store_false", help="Disable skipping to first real spoken dialogue")
    parser.add_argument("--skip-seconds", type=float, help="Explicitly skip this many seconds from the beginning")
    parser.add_argument("--min-dialogue", type=float, default=1.0, help="Minimum length for dialogue segments (default: 1.0)")
    parser.add_argument("--batch", action="store_true", help="Process all video files in the input directory")
    parser.set_defaults(skip_intro=True)
    args = parser.parse_args()

    main(
        args.input, args.output, args.silence, args.model, 
        args.skip_intro, args.skip_seconds, args.min_dialogue, args.batch
    )


