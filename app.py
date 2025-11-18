import gradio as gr
from transformers import pipeline
import time
import os

# ----- Distil-Whisper 모델 로드 (전역으로 한 번만) -----
# 다른 모델로 교체 가능:
# - "distil-whisper/distil-large-v3" (추천, 빠름)
# - "openai/whisper-large-v3" (더 정확하지만 느림)
# - "openai/whisper-turbo" (8배 빠름)
MODEL_NAME = "distil-whisper/distil-large-v3"

print(f"Loading model: {MODEL_NAME}...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,  # 30초씩 청크로 처리
    device=-1  # CPU 사용 (GPU: 0)
)
print("Model loaded successfully!")


def transcribe_streaming(audio_file, progress=gr.Progress()):
    """
    audio_file: Gradio가 넘겨주는 오디오 파일 경로 (str)
    progress: Gradio Progress tracker
    yield: 실시간으로 전사된 텍스트를 단어 단위로 스트리밍
    """
    if audio_file is None:
        yield "파일을 업로드해주세요."
        return

    start_time = time.time()

    try:
        # 초기 상태 표시
        progress(0, desc="전사 준비 중...")
        yield "🔄 전사 시작 중...\n(처음 실행 시 모델 다운로드로 2-3분 소요될 수 있습니다)"

        # 청크 단위로 처리 (30초씩)
        progress(0.3, desc="음성 분석 중...")
        yield "🔄 음성 파일 분석 중..."

        result = pipe(
            audio_file,
            return_timestamps=True,
            generate_kwargs={"language": None}  # 자동 언어 감지
        )

        progress(0.6, desc="텍스트 변환 중...")
        yield "🔄 텍스트로 변환 중..."

        # 전체 텍스트 추출
        full_text = result["text"].strip()

        if not full_text:
            yield "[전사 결과 없음]"
            return

        # ChatGPT 스타일: 단어 단위로 스트리밍 출력
        progress(0.8, desc="결과 출력 중...")
        words = full_text.split()
        current_text = ""

        for i, word in enumerate(words):
            current_text += word + " "
            yield current_text

            # 부드러운 애니메이션 (단어마다 약간의 딜레이)
            time.sleep(0.03)

            # Progress 업데이트
            progress_val = 0.8 + (0.2 * (i + 1) / len(words))
            progress(progress_val, desc=f"출력 중... ({i+1}/{len(words)} 단어)")

        # 마지막에 메타데이터 추가
        elapsed = time.time() - start_time
        final_text = current_text.strip() + f"\n\n---\n✅ 완료 | 모델: {MODEL_NAME.split('/')[-1]} | 처리 시간: {elapsed:.1f}초"
        progress(1.0, desc="완료!")
        yield final_text

    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}\n\n디버그 정보:\n- 파일: {audio_file}\n- 오류 타입: {type(e).__name__}"
        yield error_msg


# ----- Gradio UI 구성 -----
with gr.Blocks(title="pilgi — 필기를 텍스트로", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📝 pilgi — 필기를 텍스트로
        모든 음성/비디오를 텍스트로 변환합니다.

        **지원 형식**: mp3, wav, m4a, mp4, mov 등 | **다국어 자동 인식**
        """
    )

    # 파일 업로드
    audio_input = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="📎 음성/비디오 파일 업로드"
    )

    # 전사 버튼
    transcribe_btn = gr.Button("🎯 전사 시작", variant="primary", size="lg")

    # 실시간 전사 결과
    text_output = gr.Textbox(
        label="📄 전사 결과",
        lines=20,
        show_label=True,
        show_copy_button=True,  # Copy 버튼 자동 생성
        placeholder="전사 결과가 여기에 실시간으로 표시됩니다..."
    )

    # Copy All & Download 버튼
    with gr.Row():
        download_btn = gr.DownloadButton(
            label="⬇️ TXT 다운로드",
            variant="secondary"
        )

    # 이벤트 연결
    transcribe_btn.click(
        fn=transcribe_streaming,
        inputs=audio_input,
        outputs=text_output
    )

    # 다운로드 버튼 이벤트
    def prepare_download(text):
        """텍스트를 파일로 저장"""
        if not text or text.startswith("파일을") or text.startswith("🔄"):
            return None

        filename = f"transcription_{int(time.time())}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return filename

    text_output.change(
        fn=prepare_download,
        inputs=text_output,
        outputs=download_btn
    )

# Queue 활성화 (비동기 처리)
demo.queue()

if __name__ == "__main__":
    demo.launch()
