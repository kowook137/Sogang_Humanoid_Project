#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPT_TAG="20250606v2pro"
GPT_COMMIT="d7c2210da8c013e81a94bfc7b811a477c99fd506"
GPT_DIR="${GPT_SOVITS_DIR:-$HOME/GPT-SoVITS-$GPT_TAG}"
CONDA_ENV="${GPT_SOVITS_ENV:-GPTSoVits}"
BRANCH="exaone35-78b-qlora-v3"

usage() {
  cat <<'EOF'
사용법: bash busan_tts_recording_kit/lab_tts.sh COMMAND
  status          GPU·녹음·환경·학습 결과 확인 (설치 없음)
  setup           공식 GPT-SoVITS v2Pro와 사전학습 모델 자동 설치
  prepare         검수 완료 WAV를 GPT-SoVITS .list로 변환
  start           한국어 WebUI 실행
  save-audio      녹음과 매니페스트를 GitHub 브랜치에 푸시
  backup-results  학습 결과를 GitHub Release에 업로드
EOF
}

count_files() {
  local root="$1" pattern="$2"
  if [[ -d "$root" ]]; then
    find "$root" -type f -name "$pattern" | wc -l
  else
    printf '0\n'
  fi
}

status() {
  echo "===== 저장소 ====="
  printf '경로: %s\n' "$REPO_ROOT"
  printf '커밋: %s\n' "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  printf '브랜치: %s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
  git -C "$REPO_ROOT" status --short
  echo "===== GPU ====="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  else
    echo "실패: nvidia-smi가 없습니다."
  fi
  echo "===== 녹음 ====="
  printf 'WAV: %s/50\n' "$(count_files "$SCRIPT_DIR/recordings/raw" 'busan_*.wav')"
  printf '검수 완료 행: '
  awk -F, 'NR > 1 && $6 == "ready" && $7 == "true" {n++} END {print n+0}' "$SCRIPT_DIR/recording_manifest.csv"
  echo "===== GPT-SoVITS ====="
  if [[ -d "$GPT_DIR/.git" ]]; then
    printf '경로: %s\n' "$GPT_DIR"
    printf '커밋: %s\n' "$(git -C "$GPT_DIR" rev-parse HEAD)"
  else
    echo "미설치: $GPT_DIR"
  fi
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "Conda 환경: 있음 ($CONDA_ENV)"
  else
    echo "Conda 환경: 없음 ($CONDA_ENV)"
  fi
  echo "===== 학습 결과 ====="
  printf 'SoVITS 가중치: %s\n' "$(count_files "$GPT_DIR/SoVITS_weights_v2Pro" '*.pth')"
  printf 'GPT 가중치: %s\n' "$(count_files "$GPT_DIR/GPT_weights_v2Pro" '*.ckpt')"
  printf '실험 로그 파일: %s\n' "$(count_files "$GPT_DIR/logs" '*')"
}

setup() {
  command -v git >/dev/null || { echo "실패: git이 필요합니다." >&2; exit 1; }
  command -v conda >/dev/null || { echo "실패: conda가 필요합니다." >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { echo "실패: NVIDIA GPU가 보이지 않습니다." >&2; exit 1; }
  nvidia-smi >/dev/null
  if [[ ! -d "$GPT_DIR/.git" ]]; then
    git clone --depth 1 --branch "$GPT_TAG" https://github.com/RVC-Boss/GPT-SoVITS.git "$GPT_DIR"
  fi
  local actual
  actual="$(git -C "$GPT_DIR" rev-parse HEAD)"
  [[ "$actual" == "$GPT_COMMIT" ]] || { echo "실패: GPT-SoVITS 커밋 불일치: $actual" >&2; exit 1; }
  if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" python=3.10
  fi
  if [[ ! -f "$GPT_DIR/.busan_setup_complete" ]]; then
    (cd "$GPT_DIR" && conda run -n "$CONDA_ENV" bash install.sh --device CU126 --source HF)
    touch "$GPT_DIR/.busan_setup_complete"
  fi
  conda run -n "$CONDA_ENV" python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.__version__)'
  echo "설치·CUDA 검증 완료"
}

prepare() {
  python3 "$SCRIPT_DIR/prepare_gptsovits_list.py" --manifest "$SCRIPT_DIR/recording_manifest.csv" --output "$SCRIPT_DIR/busan_train.list"
}

start() {
  [[ -d "$GPT_DIR/.git" ]] || { echo "먼저 setup을 실행하세요." >&2; exit 1; }
  prepare
  cd "$GPT_DIR"
  exec conda run --no-capture-output -n "$CONDA_ENV" python webui.py ko
}

save_audio() {
  cd "$REPO_ROOT"
  [[ "$(count_files "$SCRIPT_DIR/recordings/raw" 'busan_*.wav')" -gt 0 ]] || { echo "저장할 WAV가 없습니다." >&2; exit 1; }
  git add busan_tts_recording_kit/recordings busan_tts_recording_kit/recording_manifest.csv
  [[ ! -f busan_tts_recording_kit/busan_train.list ]] || git add busan_tts_recording_kit/busan_train.list
  git commit -m "Add Busan speaker recordings and verified manifest" || true
  git push origin "$BRANCH"
}

backup_results() {
  command -v gh >/dev/null || { echo "실패: GitHub CLI(gh)가 필요합니다." >&2; exit 1; }
  gh auth status >/dev/null
  local stamp archive release_tag
  stamp="$(date +%Y%m%d-%H%M%S)"
  archive="/tmp/busan-tts-results-$stamp.tar.gz"
  release_tag="busan-tts-backup"
  local items=()
  for item in logs SoVITS_weights_v2Pro GPT_weights_v2Pro; do
    [[ -e "$GPT_DIR/$item" ]] && items+=("$item")
  done
  [[ "${#items[@]}" -gt 0 ]] || { echo "백업할 학습 결과가 없습니다." >&2; exit 1; }
  tar -C "$GPT_DIR" -czf "$archive" "${items[@]}"
  sha256sum "$archive" > "$archive.sha256"
  cd "$REPO_ROOT"
  if ! gh release view "$release_tag" >/dev/null 2>&1; then
    gh release create "$release_tag" --target "$BRANCH" --title "Busan TTS rolling backup" --notes "연구실 컴퓨터 로그아웃 대비 TTS 학습 결과"
  fi
  gh release upload "$release_tag" "$archive" "$archive.sha256" --clobber
  echo "백업 완료: $release_tag / $(basename "$archive")"
}

case "${1:-}" in
  status) status ;;
  setup) setup ;;
  prepare) prepare ;;
  start) start ;;
  save-audio) save_audio ;;
  backup-results) backup_results ;;
  *) usage; exit 2 ;;
esac
