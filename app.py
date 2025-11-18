import gradio as gr
from transformers import pipeline
import time
import os

# ----- 모델 설정 -----
# 다른 모델로 교체 가능:
# - "distil-whisper/distil-large-v3" (추천, 빠름)
# - "openai/whisper-large-v3" (더 정확하지만 느림)
# - "openai/whisper-turbo" (8배 빠름)
MODEL_NAME = "distil-whisper/distil-large-v3"

# 전역 변수: 모델 파이프라인 (처음엔 None)
pipe = None


def download_model(progress=gr.Progress()):
    """모델을 다운로드하고 로드하는 함수"""
    global pipe

    if pipe is not None:
        yield "✅ 모델이 이미 로드되어 있습니다!"
        return

    try:
        progress(0, desc="모델 다운로드 준비 중...")
        yield "🔄 모델 다운로드 시작...\n(최초 1회만, 약 1.5GB, 2-5분 소요)"

        progress(0.2, desc="Distil-Whisper 다운로드 중...")
        yield "🔄 Distil-Whisper Large v3 다운로드 중...\n(잠시만 기다려주세요)"

        # 모델 로드
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,  # 30초씩 청크로 처리
            device=-1  # CPU 사용 (GPU: 0)
        )

        progress(1.0, desc="완료!")
        yield "✅ 모델 다운로드 및 로드 완료!\n이제 음성 파일을 전사할 수 있습니다."

    except Exception as e:
        yield f"❌ 모델 로드 실패: {str(e)}\n\n다시 시도해주세요."


def transcribe_streaming(audio_file, progress=gr.Progress()):
    """
    audio_file: Gradio가 넘겨주는 오디오 파일 경로 (str)
    progress: Gradio Progress tracker
    yield: 실시간으로 전사된 텍스트를 단어 단위로 스트리밍
    """
    global pipe

    if audio_file is None:
        yield "파일을 업로드해주세요."
        return

    # 모델이 로드되지 않았으면 에러
    if pipe is None:
        yield "❌ 먼저 '모델 다운로드' 버튼을 클릭하여 모델을 로드해주세요!"
        return

    start_time = time.time()

    try:
        # 초기 상태 표시
        progress(0, desc="전사 준비 중...")
        yield "🔄 전사 시작 중..."

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

    # 모델 다운로드 섹션
    with gr.Row():
        download_model_btn = gr.Button(
            "📥 모델 다운로드 (최초 1회 필수)",
            variant="secondary",
            size="lg"
        )

    model_status = gr.Textbox(
        label="모델 상태",
        value="⚠️ 모델 미설치 - 위 버튼을 클릭하여 다운로드하세요",
        lines=3,
        interactive=False
    )

    gr.Markdown("---")

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
    # 모델 다운로드 버튼
    download_model_btn.click(
        fn=download_model,
        inputs=None,
        outputs=model_status
    )

    # 전사 버튼
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
