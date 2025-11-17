import gradio as gr
import whisper
import time

# ----- Whisper 모델 로드 (전역으로 한 번만) -----
# CPU에서 너무 느리면 "tiny"나 "base"로 바꿔봐도 됨.
MODEL_NAME = "base"
model = whisper.load_model(MODEL_NAME)


def transcribe(audio_file):
    """
    audio_file: Gradio가 넘겨주는 오디오 파일 경로 (str)
    return: (전사 텍스트, 정보 문자열)
    """
    if audio_file is None:
        return "No file uploaded.", "Please upload an audio file."

    start_time = time.time()

    # Whisper는 언어 자동 감지 지원. (language=None)
    # task="transcribe" → 음성을 텍스트로.
    result = model.transcribe(
        audio_file,
        task="transcribe",
        language=None,       # None이면 자동 감지
        verbose=False
    )

    text = result.get("text", "").strip()
    detected_lang = result.get("language", "unknown")
    elapsed = time.time() - start_time

    info = (
        f"Model: {MODEL_NAME}\n"
        f"Detected language: {detected_lang}\n"
        f"Duration: {elapsed:.1f} seconds (CPU)\n"
    )

    if not text:
        text = "[No transcription result]"

    return text, info


# ----- Gradio UI 구성 -----
with gr.Blocks(title="pilgi — Universal Transcription") as demo:
    gr.Markdown(
        """
        # 📝 pilgi — Transcribe Anything (CPU demo)
        - 업로드: **mp3 / mp4 / wav / m4a ...**
        - **모든 언어 자동 인식** (Whisper base multilingual)
        - CPU에서 돌아가는 데모라, 파일이 길면 다소 느릴 수 있어요.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Upload audio file"
        )

    with gr.Row():
        text_output = gr.Textbox(
            label="Transcription",
            lines=15,
            show_label=True
        )
        info_output = gr.Textbox(
            label="Info",
            lines=5
        )

    transcribe_btn = gr.Button("Transcribe")

    transcribe_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[text_output, info_output]
    )

# Gradio Space에서는 launch()에 server_name 등 안 넣어도 됨.
if __name__ == "__main__":
    demo.launch()
