#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
POSE_MODEL="${JETSON_POSE_PT:-${PROJECT_DIR}/openpose/models/yolo11n-pose.pt}"
CLASSIFIER="${JETSON_FALL_CLASSIFIER:-${PROJECT_DIR}/outputs/fallvision_engineered_training/best_model.pt}"
POSE_CLASSIFIER="${JETSON_POSE_CLASSIFIER:-${PROJECT_DIR}/outputs/fallvision_pose_training/best_model.pt}"
DESTINATION="${1:-${PROJECT_DIR}/outputs/sogang_fall_jetson_bundle.tar.gz}"

for required in "${POSE_MODEL}" "${CLASSIFIER}" "${POSE_CLASSIFIER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required model not found: ${required}" >&2
    exit 1
  fi
done

STAGING="$(mktemp -d)"
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/Sogang_Humanoid_Project/openpose/models"
mkdir -p "${STAGING}/Sogang_Humanoid_Project/outputs/fallvision_engineered_training"
mkdir -p "${STAGING}/Sogang_Humanoid_Project/outputs/fallvision_pose_training"
cp -a "${SCRIPT_DIR}" "${STAGING}/Sogang_Humanoid_Project/fall_detection"
cp -a "${POSE_MODEL}" "${STAGING}/Sogang_Humanoid_Project/openpose/models/yolo11n-pose.pt"
cp -a "${CLASSIFIER}" \
  "${STAGING}/Sogang_Humanoid_Project/outputs/fallvision_engineered_training/best_model.pt"
cp -a "${POSE_CLASSIFIER}" \
  "${STAGING}/Sogang_Humanoid_Project/outputs/fallvision_pose_training/best_model.pt"
mkdir -p "$(dirname -- "${DESTINATION}")"
tar -C "${STAGING}" -czf "${DESTINATION}" Sogang_Humanoid_Project
echo "bundle=${DESTINATION}"
echo "sha256=$(sha256sum "${DESTINATION}" | cut -d' ' -f1)"
