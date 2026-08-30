# Vertex AI Gemini 교사 모델 10개 파일럿 매뉴얼

## 원칙

- L4 GPU VM은 중지한다. Gemini 호출에는 GPU VM이 필요하지 않다.
- 무료 Cloud Shell에서 실행한다.
- 처음에는 모델별 10개만 생성한다.
- 결과를 사람이 읽기 전에는 100개로 확대하지 않는다.
- 자동검사 통과는 자연스러움의 최종 승인을 의미하지 않는다.

## 1. Cloud Shell 열기

Google Cloud 콘솔 오른쪽 위의 터미널 아이콘을 눌러 Cloud Shell을 연다.
프로젝트가 다음 값인지 확인한다.

```bash
gcloud config get-value project
```

다른 값이면 설정한다.

```bash
gcloud config set project project-a965291e-1224-4ea7-bf1
```

## 2. Vertex AI API 활성화

```bash
gcloud services enable aiplatform.googleapis.com
```

## 3. 파일과 Python 환경 준비

```bash
mkdir -p ~/vertex-teacher/data
cd ~/vertex-teacher

BASE="https://raw.githubusercontent.com/kowook137/Sogang_Humanoid_Project/exaone35-78b-qlora-v3/exaone_finetuning"

wget -qO generate_vertex_gold_pilot.py "$BASE/generate_vertex_gold_pilot.py"
wget -qO gold_review.py "$BASE/gold_review.py"
wget -qO data/questions.jsonl "$BASE/data/gold/gyeongsang_batch_001_questions.jsonl"
wget -qO requirements.txt "$BASE/requirements-vertex-teacher.txt"

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
wc -l data/questions.jsonl
```

마지막 값은 `100`이어야 한다.

## 4. Flash 10개 생성

```bash
.venv/bin/python generate_vertex_gold_pilot.py \
  --input data/questions.jsonl \
  --output data/flash_accepted.jsonl \
  --rejected data/flash_rejected.jsonl \
  --model gemini-2.5-flash \
  --limit 10
```

## 5. Pro 10개 생성

```bash
.venv/bin/python generate_vertex_gold_pilot.py \
  --input data/questions.jsonl \
  --output data/pro_accepted.jsonl \
  --rejected data/pro_rejected.jsonl \
  --model gemini-2.5-pro \
  --limit 10
```

## 6. 자동검사 결과 확인

```bash
wc -l data/flash_accepted.jsonl data/flash_rejected.jsonl
wc -l data/pro_accepted.jsonl data/pro_rejected.jsonl
```

각 모델에서 accepted와 rejected의 합이 10이어야 한다. 자동 거절된 결과도
모델 비교에 필요하므로 삭제하지 않는다.

## 7. 검토용 CSV 생성

```bash
cat data/flash_accepted.jsonl data/flash_rejected.jsonl > data/flash_all.jsonl
cat data/pro_accepted.jsonl data/pro_rejected.jsonl > data/pro_all.jsonl

.venv/bin/python gold_review.py prepare \
  --input data/flash_all.jsonl \
  --output ~/gemini_flash_pilot.csv \
  --limit 10

.venv/bin/python gold_review.py prepare \
  --input data/pro_all.jsonl \
  --output ~/gemini_pro_pilot.csv \
  --limit 10
```

Cloud Shell의 파일 다운로드 메뉴에서 다음 두 경로를 내려받는다.

```text
/home/hanseo501/gemini_flash_pilot.csv
/home/hanseo501/gemini_pro_pilot.csv
```

## 8. 사람 평가

각 질문에 대해 다음 항목을 확인한다.

1. 질문에 제대로 답했는가
2. 의미와 안전 정보가 유지됐는가
3. 존댓말인가
4. 부산·경남 화자가 자연스럽게 느끼는가
5. 과장된 사투리나 특정 어미 반복이 없는가

10개 결과를 확인하기 전에는 추가 생성이나 학습을 시작하지 않는다.

## 9. 프롬프트 개선판으로 같은 10개 재시험

첫 파일럿과 결과가 섞이지 않도록 새 파일명을 사용한다.

```bash
cd ~/vertex-teacher

wget -qO generate_vertex_gold_pilot.py \
  "https://raw.githubusercontent.com/kowook137/Sogang_Humanoid_Project/exaone35-78b-qlora-v3/exaone_finetuning/generate_vertex_gold_pilot.py"

.venv/bin/python generate_vertex_gold_pilot.py \
  --input data/questions.jsonl \
  --output data/flash_v2_accepted.jsonl \
  --rejected data/flash_v2_rejected.jsonl \
  --model gemini-2.5-flash \
  --limit 10

wc -l data/flash_v2_accepted.jsonl data/flash_v2_rejected.jsonl

cat data/flash_v2_accepted.jsonl data/flash_v2_rejected.jsonl > data/flash_v2_all.jsonl
.venv/bin/python gold_review.py prepare \
  --input data/flash_v2_all.jsonl \
  --output ~/gemini_flash_pilot_v2.csv \
  --limit 10
```

다운로드 경로는 `/home/hanseo501/gemini_flash_pilot_v2.csv`이다. 첫 파일럿 CSV와
나란히 놓고 의미 보존, 자연스러움, 사투리 강도를 비교한다.
