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
                    "content": "You are an expert at creating concise, informative study notes from video transcripts. Your notes should be well-structured with major headings and subheadings. Provide the note in Korean.",
                },
                {
                    "role": "user",
                    "content": f"""다음 트랜스크립트를 바탕으로 학습 노트를 한국어로 작성해주세요. 아래 지침을 따라주세요:

1. 노트 시작 부분에 영상 전체에 대한 간단한 요약 문장과 이 영상에서 배울 수 있는 점을 작성하세요.
2. 마크다운 형식을 사용하세요.
3. 각 소제목 아래에 해당 내용의 유튜브 타임스탬프 링크를 포함하세요.
4. 타임스탬프 링크는 다음 형식을 사용하세요: [▶️](https://youtu.be/{video_id}?t=HH:MM:SS)
   HH:MM:SS를 트랜스크립트에 나타난 정확한 타임스탬프로 교체하세요 (예: 00:03:18).
5. 타임스탬프 링크는 각 소제목 바로 다음에 배치하세요.
6. 트랜스크립트에 나온 정확한 타임스탬프를 사용하고, 시간을 계산하거나 변환하지 마세요.
7. 각 소제목에 해당 섹션의 영상 내용을 서술형으로 상세히 서술하세요.
8. 구체적인 정보를 포함한 문장을 작성하는 데 집중하세요. 예를 들어, 영상에서 주제 A에 대해 설명한다면, "영상에서 A에 대해 설명합니다"라고만 쓰지 마세요. 대신, 영상에서 A를 뭐라고 설명했는지를 작성하세요.
9. 주어진 트랜스크립트가 프로그램에 의해 생성되었을 수 있으며, 발음이 비슷하지만 잘못된 단어가 포함될 수 있습니다. 잘못된 것 같은 단어를 발견하면 문맥에 맞는 비슷한 발음의 단어로 해석하세요.
10. 다음 형식에 맞춰서 학습 노트를 작성하세요:

# [제목]
- [영상 전체 요약과 이 영상에서 배울 수 있는 점]

## 1. [소제목]
[▶️](https://youtu.be/{video_id}?t=HH:MM:SS)

- [이 섹션의 내용을 완전한 문장으로 설명]
- [필요한 만큼 불릿 포인트 사용]
- [영상을 보지 않은 사람도 내용을 알 수 있을 만큼 자세히 작성]
- [한 불릿에 한 문장씩 작성]

.. (필요한 만큼 소제목 반복하여 사용.) .. 

## [숫자]. [소제목]
[▶️](https://youtu.be/{video_id}?t=HH:MM:SS)
- [위와 마찬가지로 학습 노트 작성]
- [최대한 많은 정보를 담을 수 있도록 자세히 작성]


다음은 트랜스크립트입니다:

{transcript_content}""",
                },
            ],
            max_tokens=5000,
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
        output_path = os.path.join(output_folder, f"{timestamp}.jpg")
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

    asset_folder = os.path.join(frame_folder, str(hash(summary_path)))
    os.makedirs(asset_folder, exist_ok=True)

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

        frame_filename = f"{timestamp}.jpg"
        frame_path = os.path.join(asset_folder, frame_filename)
        if extract_frame(video_path, timestamp, asset_folder):
            return f"\n[![Frame at {timestamp}s](/{frame_path})]({url})\n"
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
    youtube_url = input("Enter the YouTube video URL: ")
    result = summarize_youtube_video(youtube_url)
    if result:
        print(f"Summary created successfully. You can find it at: {result}")
    else:
        print("Failed to create summary.")
