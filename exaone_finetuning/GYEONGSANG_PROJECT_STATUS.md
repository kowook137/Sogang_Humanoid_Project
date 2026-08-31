# 경상도 대화 모델 현재 상태와 다음 작업

최종 갱신: 2026-09-01 (KST)

## 현재 결정

- 최종 모델은 EXAONE 3.5 7.8B Instruct이다.
- Gemini는 학습 후보를 만드는 교사 모델로만 사용한다.
- 최종 데이터는 사람이 승인하거나 직접 수정한 문장만 사용한다.
- 프로젝트 요구사항의 정본은 `GYEONGSANG_MODEL_SPEC.md`이다.
- 수량과 분포는 `dataset_plan.json`, 성공 기준은 `evaluation_plan.json`을 따른다.

## 완료된 작업

- EXAONE 3.5 7.8B QLoRA 실행 환경과 학습·비교 경로를 확인했다.
- 기존 소량 LoRA가 일부 사투리를 출력했으나 일반 대화 전체에서 안정적이지 않음을
  확인했다.
- EXAONE을 교사 모델로 사용한 자동 후보 생성은 품질이 낮아 중단했다.
- Vertex AI Gemini Flash 파일럿 v1~v3를 수행하고 동시 생성, 말투 강도, 행 단위
  자동 거절의 문제를 확인했다.
- 표준 답변 생성과 사투리 변환을 두 호출로 분리했다.
- 자동검사를 후보 단위로 바꾸고 검토 CSV에 후보별 사유를 기록하도록 구현했다.
- Gemini 2.5 Pro 10개 파일럿에서 모든 질문에 사용 가능한 후보가 최소 하나 존재했다.
- Pro 파일럿에 대해 8개 직접 선택, 2개 수정 권장안을 작성했다. 사용자의 최종 검수
  확정과 export는 아직 수행하지 않았다.

## 바로 다음 작업

1. `gemini_pro_candidate_pilot.csv`에 최종 `decision`과 필요한 `edited_answer`를 기록한다.
2. `gold_review.py validate`와 `gold_review.py export`로 승인 데이터만 내보낸다.
3. `dataset_plan.json`의 `pilot_100` 비율에 맞춰 단일·다중 대화 시나리오 100개를 만든다.
4. Gemini Pro로 후보를 생성하고 부산·경남 화자가 전량 검수한다.
5. `validate_conversation_dataset.py --strict`로 구조·중복·검수 여부를 검사한다.
6. 100개에서 직접 채택 또는 수정 가능 비율이 80% 이상이면 1,000개로 확대한다.
7. 학습 데이터와 별도로 비공개 평가 200개를 제작한다.
8. EXAONE QLoRA 후 베이스 모델과 어댑터를 동일 평가 세트로 비교한다.

## 재개 명령

```bash
cd /home/hanseo501/projects/Sogang_Humanoid_Project
git status --short --branch
sed -n '1,240p' exaone_finetuning/GYEONGSANG_MODEL_SPEC.md
cat exaone_finetuning/dataset_plan.json
cat exaone_finetuning/evaluation_plan.json
sed -n '1,220p' exaone_finetuning/GYEONGSANG_PROJECT_STATUS.md
```

새 대화에서는 다음과 같이 요청한다.

> 경상도 모델 정본 명세와 프로젝트 상태, 데이터·평가 계획, Git 상태를 읽고 다음 작업을
> 이어가줘. 기존 미커밋 변경은 덮어쓰지 마.
