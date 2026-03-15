#!/usr/bin/env bash

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dataset_name> [dataset_name ...]" >&2
  exit 1
fi

PROJECT_ROOT="/workspaces/slam_project"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/output"
WS_DIR="${PROJECT_ROOT}/ros2_ws"

source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"
set -u

cleanup() {
  local pids=("${OPENVINS_PID:-}" "${BRIDGE_PID:-}" "${RECORD_PID:-}")
  local pid
  local signal

  for signal in INT TERM KILL; do
    for pid in "${pids[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "-${signal}" "${pid}" 2>/dev/null || true
      fi
    done
    sleep 1
  done

  for pid in "${pids[@]}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT

cleanup_stale_processes() {
  pkill -INT -f '/workspaces/slam_project/ros2_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf' 2>/dev/null || true
  pkill -INT -f '/workspaces/slam_project/ros2_ws/install/visual_slam_bridge/lib/visual_slam_bridge/novel_pointcloud_bridge' 2>/dev/null || true
  pkill -INT -f 'ros2 bag record' 2>/dev/null || true
  pkill -INT -f 'ros2 bag play' 2>/dev/null || true
  sleep 2
  pkill -TERM -f '/workspaces/slam_project/ros2_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf' 2>/dev/null || true
  pkill -TERM -f '/workspaces/slam_project/ros2_ws/install/visual_slam_bridge/lib/visual_slam_bridge/novel_pointcloud_bridge' 2>/dev/null || true
  pkill -TERM -f 'ros2 bag record' 2>/dev/null || true
  pkill -TERM -f 'ros2 bag play' 2>/dev/null || true
  sleep 2
  pkill -KILL -f '/workspaces/slam_project/ros2_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf' 2>/dev/null || true
  pkill -KILL -f '/workspaces/slam_project/ros2_ws/install/visual_slam_bridge/lib/visual_slam_bridge/novel_pointcloud_bridge' 2>/dev/null || true
  pkill -KILL -f 'ros2 bag record' 2>/dev/null || true
  pkill -KILL -f 'ros2 bag play' 2>/dev/null || true
}

run_dataset() {
  local dataset="$1"
  local src_bag="${DATA_DIR}/${dataset}.bag"
  local ros2_bag_dir="${DATA_DIR}/${dataset}_ros2_humble"
  local timestamp
  local run_dir
  local record_dir

  if [[ ! -f "${src_bag}" ]]; then
    echo "missing source bag: ${src_bag}" >&2
    return 1
  fi

  if [[ ! -d "${ros2_bag_dir}" ]]; then
    echo "[${dataset}] converting rosbag1 to rosbag2 humble format"
    rosbags-convert \
      --src "${src_bag}" \
      --dst "${ros2_bag_dir}" \
      --dst-version 8 \
      --dst-typestore ros2_humble \
      --include-topic /d455/imu /d455/color/image_raw
  else
    echo "[${dataset}] reusing existing ros2 bag: ${ros2_bag_dir}"
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="${OUTPUT_DIR}/${dataset}_openvins_run_${timestamp}"
  record_dir="${run_dir}/${dataset}_visual_slam_bag"
  mkdir -p "${run_dir}"

  echo "[${dataset}] writing outputs under ${run_dir}"

  cleanup_stale_processes

  "${WS_DIR}/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf" \
    /workspaces/slam_project/ros2_ws/src/open_vins/config/rs_d455/estimator_config.yaml \
    --ros-args \
    -r __ns:=/ov_msckf \
    -p verbosity:=INFO \
    >"${run_dir}/openvins.log" 2>&1 &
  OPENVINS_PID=$!
  sleep 5

  "${WS_DIR}/install/visual_slam_bridge/lib/visual_slam_bridge/novel_pointcloud_bridge" \
    >"${run_dir}/bridge.log" 2>&1 &
  BRIDGE_PID=$!
  sleep 3

  ros2 bag record \
    --max-cache-size 0 \
    --use-sim-time \
    -o "${record_dir}" \
    /visual_slam/tracking/odometry \
    /visual_slam/mapping/point_cloud \
    >"${run_dir}/record.log" 2>&1 &
  RECORD_PID=$!
  sleep 5

  echo "[${dataset}] playing bag ${ros2_bag_dir}"
  ros2 bag play "${ros2_bag_dir}" --read-ahead-queue-size 5000 --clock 100 \
    >"${run_dir}/play.log" 2>&1

  sleep 5
  kill -INT "${RECORD_PID}" 2>/dev/null || true
  wait "${RECORD_PID}" || true
  unset RECORD_PID

  kill -INT "${BRIDGE_PID}" 2>/dev/null || true
  sleep 1
  kill -TERM "${BRIDGE_PID}" 2>/dev/null || true
  sleep 1
  kill -KILL "${BRIDGE_PID}" 2>/dev/null || true
  wait "${BRIDGE_PID}" || true
  unset BRIDGE_PID

  kill -INT "${OPENVINS_PID}" 2>/dev/null || true
  sleep 1
  kill -TERM "${OPENVINS_PID}" 2>/dev/null || true
  sleep 1
  kill -KILL "${OPENVINS_PID}" 2>/dev/null || true
  wait "${OPENVINS_PID}" || true
  unset OPENVINS_PID

  echo "[${dataset}] finished"
}

for dataset in "$@"; do
  run_dataset "${dataset}"
done
