---
title: Pilgi
emoji: 📝
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: 필기를 텍스트로 - Universal Transcription
---

# 📝 pilgi — 필기를 텍스트로

**모든 음성/비디오를 텍스트로 변환하는 웹 앱**

## ✨ 주요 기능

- 🎙️ **실시간 스트리밍 전사**: ChatGPT 스타일로 단어 단위 실시간 출력
- 🌍 **다국어 자동 인식**: 한국어, 영어, 일본어, 중국어 등 99개 언어 지원
- ⚡ **초고속 처리**: Faster-Whisper (CTranslate2) - 3-4배 빠른 처리
- 📋 **간편한 복사/다운로드**: Copy 버튼과 TXT 다운로드 지원
- 🎵 **다양한 형식 지원**: mp3, wav, m4a, mp4, mov 등

## 🚀 사용 방법

1. 음성/비디오 파일 업로드 또는 마이크로 직접 녹음
2. "🎯 전사 시작" 버튼 클릭
3. 실시간으로 전사 결과 확인
4. Copy 버튼으로 복사하거나 TXT로 다운로드

## 🔧 기술 스택

- **모델**: Faster-Whisper Tiny (SYSTRAN) - 39M params, INT8 양자화
- **추론 엔진**: CTranslate2 (3-4배 빠른 CPU 최적화)
- **프레임워크**: Gradio 5.0+
- **처리 방식**: 실시간 스트리밍 출력
- **환경**: HuggingFace Spaces (CPU)

## 🎯 로드맵

- [x] 음성 전사 (프로토타입)
- [ ] 비디오 자동 추출 및 전사
- [ ] 이미지 OCR (손글씨 인식)
- [ ] PDF 전사
- [ ] GoodNotes 파일 지원

## 📄 라이센스

MIT License

---

**pilgi** = 한국어로 "필기" (Note-taking)
