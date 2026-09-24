# 부산 TTS 녹음 키트

이 키트는 GPT-SoVITS 한국어 음성 합성 실험을 위한 녹음 대본과 파일 manifest입니다. 개인 녹음은 아직 포함하지 않습니다.

## 녹음할 파일

1. 기준 음성 후보 중 **하나**를 골라 3~10초 녹음하고, 실제 발화와 정확히 일치하는 대본을 함께 저장합니다.
2. `recording_sentences.md`의 50개 문장을 각기 별도 WAV 파일로 녹음합니다. 대본 문구를 임의로 사투리로 고치지 말고 적힌 그대로 읽습니다. 평소 부산 억양으로 편하게 말하고 사투리를 과장하지 마세요.
3. `recording_manifest.csv`의 파일명과 대본을 그대로 맞춥니다. 녹음을 듣고 실제 발화와 대본이 일치함을 확인하기 전에는 `transcript_verified`를 `true`로 바꾸지 않습니다.
4. 녹음 방법과 권장 음질은 `RECORDING_GUIDE.md`에 있습니다. 원본을 덮어쓰지 마세요.

각 문장은 `recordings/raw/busan_001.wav`부터 `busan_050.wav`까지 저장합니다. 기준 음성은 `recordings/reference/busan_reference.wav`와 실제 발화 대본 `busan_reference.txt`를 사용합니다.

## 부산말 비교 문장

`comparison_texts.csv`에는 표준어 10개와 부산말 10개가 있습니다. 말투가 자연스럽지 않으면 합성 전에 화자가 검토하고 대본과 녹음을 함께 바로잡습니다.

## 다음 단계

GPU 컴퓨터에서 기준 음성 zero-shot 합성 → 50개 녹음의 잡음·전사 검수 → GPT-SoVITS few-shot 미세조정 → 비교 문장 20개 합성 순서로 진행합니다. 아직 개인 음성 녹음이나 학습 결과는 없습니다.

저장소의 `LAB_GPU_HANDOFF.md`에는 GPU 컴퓨터의 우선 점검 지침이 있습니다. 작업 재개 시 Codex 대화는 `codex resume --all`로 찾을 수 있습니다.
