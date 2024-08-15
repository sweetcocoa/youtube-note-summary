from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
)
import markdown
import os
from youtube_summarizer import summarize_youtube_video

app = Flask(__name__)

SUMMARY_DIR = "summaries"
FRAMES_DIR = "frames"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def get_processed_links():
    summaries = []
    for filename in os.listdir(SUMMARY_DIR):
        if filename.endswith("_summary_with_images.md"):
            video_id = filename.split("_")[0]
            summaries.append(
                {
                    "video_id": video_id,
                    "filename": filename,
                    "url": video_id,
                }
            )
    print(f"Processed summaries: {summaries}")  # Debug print
    return summaries


@app.route("/", methods=["GET", "POST"])
def index():
    summaries = get_processed_links()
    print(f"Rendering index with summaries: {summaries}")  # Debug print
    if request.method == "POST":
        youtube_url = request.form["youtube_url"]
        use_whisper = request.form.get("use_whisper") == "on"
        summary_path = summarize_youtube_video(youtube_url, use_whisper)

        if summary_path:
            # Move the summary file to the summaries directory
            filename = os.path.basename(summary_path)
            new_path = os.path.join(SUMMARY_DIR, filename)
            os.rename(summary_path, new_path)

            return redirect(url_for("view_summary", filename=filename))
        else:
            return render_template(
                "index.html", error="Failed to create summary.", summaries=summaries
            )

    return render_template("index.html", summaries=summaries)


@app.route("/summary/<filename>")
def view_summary(filename):
    summary_path = os.path.join(SUMMARY_DIR, filename)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            content = file.read()
        html_content = markdown.markdown(content)
        summaries = get_processed_links()
        print(
            f"Rendering summary with content length: {len(html_content)}, summaries: {summaries}"
        )  # Debug print
        return render_template(
            "summary.html", summary=html_content, summaries=summaries
        )
    else:
        return "Summary not found", 404


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
