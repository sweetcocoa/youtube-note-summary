import os
import yt_dlp
import whisper
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
import cv2
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def download_youtube_video(url):
    try:
        # First, download the video
        video_ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(video_ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_id = info["id"]

        print(f"Video downloaded successfully to: {video_path}")

        # Now, download audio separately
        audio_ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(audio_ydl_opts) as ydl:
            ydl.download([url])
            audio_path = os.path.splitext(video_path)[0] + ".wav"

        print(f"Audio extracted successfully to: {audio_path}")

        return video_id, video_path, audio_path
    except Exception as e:
        print(f"An error occurred while downloading the video: {str(e)}")
        return None, None, None


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def download_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=("ko", "en,")
        )

        transcript_path = os.path.join(
            "downloads", f"{video_id}_youtube_transcript.txt"
        )
        with open(transcript_path, "w", encoding="utf-8") as f:
            for entry in transcript:
                timestamp = format_time(int(entry["start"]))
                f.write(f"[{timestamp}] {entry['text']}\n")

        print(f"YouTube transcript downloaded successfully to: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"An error occurred while downloading the YouTube transcript: {str(e)}")
        return None


def transcribe_with_whisper(audio_path):
    try:
        print("Transcribing audio with Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        transcript_path = audio_path.rsplit(".", 1)[0] + "_whisper_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                timestamp = format_time(int(segment["start"]))
                f.write(f"[{timestamp}] {segment['text']}\n")

        print(f"Whisper transcript saved successfully to: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"An error occurred while transcribing with Whisper: {str(e)}")
        return None


def create_summary(transcript, video_id):
    try:
        with open(transcript, "r", encoding="utf-8") as file:
            transcript_content = file.read()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating concise, informative study notes from video transcripts. Your notes should be well-structured with major headings and subheadings. Provide the summary in Korean.",
                },
                {
                    "role": "user",
                    "content": f"""Please create summary notes in Korean from the following transcript. Follow these guidelines:

1. At the beginning of the notes, write a brief summary sentence about the entire video and what can be learned from it.
2. Organize the notes with main topics and subtopics.
3. Use markdown formatting for headings and subheadings.
4. For each subheading, include a timestamped YouTube link that corresponds to the content of that subheading.
5. Use the following format for the timestamped links: [▶️](https://youtu.be/{video_id}?t=HH:MM:SS)
   Replace HH:MM:SS with the exact timestamp as it appears in the transcript (e.g., 00:03:18).
6. Place the timestamped link immediately after each subheading.
7. Ensure you use the exact timestamp from the transcript, do not attempt to calculate or convert the time.
8. When summarizing, use complete sentences with proper predicates rather than short phrases or bullet points. Provide detailed explanations instead of brief answers.

Here's the transcript:

{transcript_content}""",
                },
            ],
            max_tokens=3000,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while creating the summary: {str(e)}")
        return None


def fix_youtube_links(summary):
    def time_to_seconds(time_str):
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        else:
            return int(time_str)

    def replace_time(match):
        video_id = match.group(1)
        time_str = match.group(2)
        seconds = time_to_seconds(time_str)
        return f"https://youtu.be/{video_id}?t={seconds}"

    pattern = r"https://youtu\.be/([^?]+)\?t=([^)]+)"
    return re.sub(pattern, replace_time, summary)


def extract_frame(video_path, timestamp, output_folder):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, image = cap.read()
    if success:
        output_path = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(output_path, image)
        cap.release()
        return output_path
    cap.release()
    return None


def add_thumbnails(summary_path, video_path):
    with open(summary_path, "r", encoding="utf-8") as file:
        content = file.read()

    frame_folder = "frames"
    os.makedirs(frame_folder, exist_ok=True)

    def replace_link(match):
        url = match.group(1)
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        if "t" in query_params:
            timestamp_str = query_params["t"][0]
            timestamp_str = re.sub(r"\D", "", timestamp_str)
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                print(f"Warning: Invalid timestamp in URL: {url}")
                return match.group(0)
        else:
            print(f"Warning: No timestamp found in URL: {url}")
            return match.group(0)

        frame_filename = f"frame_{timestamp}.jpg"
        frame_path = os.path.join(frame_folder, frame_filename)
        if extract_frame(video_path, timestamp, frame_folder):
            return f"\n[![Frame at {timestamp}s](/{frame_path})]({url})"
        return match.group(0)

    pattern = r"\[▶️\]\((https://youtu\.be/[^)]+)\)"
    modified_content = re.sub(pattern, replace_link, content)

    output_path = os.path.splitext(summary_path)[0] + "_with_images.md"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(modified_content)

    print(f"Enhanced summary saved to: {output_path}")
    return output_path


def summarize_youtube_video(url, use_whisper=True):
    # Step 1: Download and transcribe the YouTube video
    video_id, video_path, audio_path = download_youtube_video(url)
    if not video_id or not video_path or not audio_path:
        return None

    if use_whisper:
        transcript_path = transcribe_with_whisper(audio_path)
    else:
        transcript_path = download_youtube_transcript(video_id)

    if not transcript_path:
        print("Failed to get transcript. Trying the other method as fallback.")
        if use_whisper:
            transcript_path = download_youtube_transcript(video_id)
        else:
            transcript_path = transcribe_with_whisper(audio_path)

        if not transcript_path:
            print("Both transcription methods failed.")
            return None
    # Step 2: Generate a summary note based on the transcript
    summary = create_summary(transcript_path, video_id)
    if not summary:
        return None

    summary_path = os.path.splitext(video_path)[0] + "_summary.md"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(summary)

    # Step 3: Edit the generated summary notes
    fixed_summary = fix_youtube_links(summary)
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(fixed_summary)

    # Step 4: Add thumbnails to the summary
    final_summary_path = add_thumbnails(summary_path, video_path)

    return final_summary_path


if __name__ == "__main__":
    xx = "/home/sweetcocoa/wsl/workspace/sweet-youtube-summary/downloads/[주식] 드디어 배운 '재무제표' 읽는법(홍진경,슈카) [공부왕찐천재]_summary.md"
    yy = "/home/sweetcocoa/wsl/workspace/sweet-youtube-summary/downloads/[주식] 드디어 배운 '재무제표' 읽는법(홍진경,슈카) [공부왕찐천재].mp4"
    add_thumbnails(xx, yy)

    # youtube_url = input("Enter the YouTube video URL: ")
    # result = summarize_youtube_video(youtube_url)
    # if result:
    #     print(f"Summary created successfully. You can find it at: {result}")
    # else:
    #     print("Failed to create summary.")
