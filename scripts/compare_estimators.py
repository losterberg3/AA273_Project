#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Trajectory:
    name: str
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation


def load_ground_truth_csv(path: Path) -> Trajectory:
    times = []
    positions = []
    quats = []

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(int(row["#time(ns)"]) * 1e-9)
            positions.append([float(row["px"]), float(row["py"]), float(row["pz"])])
            quats.append(
                [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ]
            )

    return Trajectory(
        name="ground_truth",
        times=np.asarray(times, dtype=float),
        positions=np.asarray(positions, dtype=float),
        rotations=Rotation.from_quat(np.asarray(quats, dtype=float)),
    )


def read_metadata_topic(metadata_path: Path) -> tuple[str, str]:
    metadata = yaml.safe_load(metadata_path.read_text())
    topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]

    for entry in topics:
        topic = entry["topic_metadata"]["name"]
        msg_type = entry["topic_metadata"]["type"]
        if msg_type in {"nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped"}:
            return topic, msg_type

    raise RuntimeError(f"No supported odometry topic found in {metadata_path}")


def load_bag_trajectory(name: str, bag_dir: Path) -> Trajectory:
    topic_name, msg_type_name = read_metadata_topic(bag_dir / "metadata.yaml")
    msg_type = get_message(msg_type_name)

    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_dir), storage_id="sqlite3"), ConverterOptions("", ""))

    times = []
    positions = []
    quats = []

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)
        if msg_type_name == "nav_msgs/msg/Odometry":
            stamp = msg.header.stamp
            pose = msg.pose.pose
        else:
            stamp = msg.header.stamp
            pose = msg.pose

        times.append(stamp.sec + stamp.nanosec * 1e-9)
        positions.append([pose.position.x, pose.position.y, pose.position.z])
        quats.append([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

    if not times:
        raise RuntimeError(f"No odometry samples found for {name} in {bag_dir}")

    return Trajectory(
        name=name,
        times=np.asarray(times, dtype=float),
        positions=np.asarray(positions, dtype=float),
        rotations=Rotation.from_quat(np.asarray(quats, dtype=float)),
    )


def interpolate_ground_truth(gt: Trajectory, query_times: np.ndarray) -> tuple[np.ndarray, Rotation]:
    mask = (query_times >= gt.times[0]) & (query_times <= gt.times[-1])
    valid_times = query_times[mask]
    if valid_times.size < 2:
        raise RuntimeError("Insufficient overlap with ground truth for interpolation")

    interp_positions = np.column_stack(
        [np.interp(valid_times, gt.times, gt.positions[:, axis]) for axis in range(3)]
    )
    slerp = Slerp(gt.times, gt.rotations)
    interp_rotations = slerp(valid_times)
    return mask, interp_positions, interp_rotations


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_zero = src - src_center
    dst_zero = dst - dst_center
    cov = src_zero.T @ dst_zero
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = dst_center - src_center @ rot.T
    return rot, trans


def nearest_indices(times: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(times, target_times)
    idx = np.clip(idx, 0, len(times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(times) - 1)
    choose_prev = np.abs(times[prev_idx] - target_times) < np.abs(times[idx] - target_times)
    nearest = np.where(choose_prev, prev_idx, idx)
    return nearest, np.abs(times[nearest] - target_times)


def compute_rpe(times: np.ndarray, est_pos: np.ndarray, est_rot: Rotation, gt_pos: np.ndarray, gt_rot: Rotation) -> tuple[float, float]:
    delta = 1.0
    target_times = times + delta
    pair_idx, time_error = nearest_indices(times, target_times)
    valid = (pair_idx > np.arange(len(times))) & (time_error < 0.05)

    if not np.any(valid):
        return float("nan"), float("nan")

    i0 = np.arange(len(times))[valid]
    i1 = pair_idx[valid]

    est_delta_pos = est_pos[i1] - est_pos[i0]
    gt_delta_pos = gt_pos[i1] - gt_pos[i0]
    trans_err = np.linalg.norm(est_delta_pos - gt_delta_pos, axis=1)

    est_rel = est_rot[i0].inv() * est_rot[i1]
    gt_rel = gt_rot[i0].inv() * gt_rot[i1]
    rot_err = (gt_rel * est_rel.inv()).magnitude()

    return float(np.sqrt(np.mean(trans_err**2))), float(np.sqrt(np.mean(np.degrees(rot_err) ** 2)))


def summarize_trajectory(est: Trajectory, gt: Trajectory) -> dict:
    mask, gt_pos_interp, gt_rot_interp = interpolate_ground_truth(gt, est.times)
    est_times = est.times[mask]
    est_pos = est.positions[mask]
    est_rot = est.rotations[mask]

    rot_align, trans_align = estimate_rigid_transform(est_pos, gt_pos_interp)
    rot_align_obj = Rotation.from_matrix(rot_align)
    est_pos_aligned = est_pos @ rot_align.T + trans_align
    est_rot_aligned = rot_align_obj * est_rot

    pos_err = np.linalg.norm(est_pos_aligned - gt_pos_interp, axis=1)
    rot_err_deg = np.degrees((gt_rot_interp * est_rot_aligned.inv()).magnitude())

    rpe_trans_rmse, rpe_rot_rmse_deg = compute_rpe(
        est_times, est_pos_aligned, est_rot_aligned, gt_pos_interp, gt_rot_interp
    )

    metrics = {
        "estimator": est.name,
        "samples": int(len(est_times)),
        "start_time_s": float(est_times[0]),
        "end_time_s": float(est_times[-1]),
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_mae_m": float(np.mean(pos_err)),
        "position_max_m": float(np.max(pos_err)),
        "final_position_error_m": float(pos_err[-1]),
        "orientation_rmse_deg": float(np.sqrt(np.mean(rot_err_deg**2))),
        "orientation_mae_deg": float(np.mean(rot_err_deg)),
        "orientation_max_deg": float(np.max(rot_err_deg)),
        "rpe_translation_rmse_m_1s": rpe_trans_rmse,
        "rpe_rotation_rmse_deg_1s": rpe_rot_rmse_deg,
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(est_pos_aligned, axis=0), axis=1))),
        "gt_path_length_m": float(np.sum(np.linalg.norm(np.diff(gt_pos_interp, axis=0), axis=1))),
    }

    return {
        "metrics": metrics,
        "times": est_times,
        "time_from_start": est_times - est_times[0],
        "gt_positions": gt_pos_interp,
        "gt_rotations": gt_rot_interp,
        "est_positions": est_pos_aligned,
        "est_rotations": est_rot_aligned,
        "position_error": pos_err,
        "orientation_error_deg": rot_err_deg,
    }


def write_metrics_csv(results: list[dict], output_dir: Path) -> None:
    fieldnames = list(results[0]["metrics"].keys())
    with (output_dir / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result["metrics"])


def write_aligned_trajectories(results: list[dict], output_dir: Path) -> None:
    for result in results:
        name = result["metrics"]["estimator"]
        with (output_dir / f"{name}_aligned_trajectory.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s",
                    "time_from_start_s",
                    "est_x",
                    "est_y",
                    "est_z",
                    "gt_x",
                    "gt_y",
                    "gt_z",
                    "position_error_m",
                    "orientation_error_deg",
                ]
            )
            for idx in range(len(result["times"])):
                writer.writerow(
                    [
                        f"{result['times'][idx]:.9f}",
                        f"{result['time_from_start'][idx]:.9f}",
                        *[f"{v:.6f}" for v in result["est_positions"][idx]],
                        *[f"{v:.6f}" for v in result["gt_positions"][idx]],
                        f"{result['position_error'][idx]:.6f}",
                        f"{result['orientation_error_deg'][idx]:.6f}",
                    ]
                )


def write_report(results: list[dict], output_dir: Path) -> None:
    sorted_results = sorted(results, key=lambda item: item["metrics"]["position_rmse_m"])
    lines = [
        "# Estimator Comparison",
        "",
        "Trajectories were rigidly aligned to the ground truth with a rotation + translation fit before scoring.",
        "",
        "| Estimator | Pos RMSE (m) | Pos MAE (m) | Final Pos Err (m) | Ori RMSE (deg) | RPE 1s Pos RMSE (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in sorted_results:
        metrics = result["metrics"]
        lines.append(
            f"| {metrics['estimator']} | {metrics['position_rmse_m']:.4f} | {metrics['position_mae_m']:.4f} | "
            f"{metrics['final_position_error_m']:.4f} | {metrics['orientation_rmse_deg']:.3f} | "
            f"{metrics['rpe_translation_rmse_m_1s']:.4f} |"
        )

    best = sorted_results[0]["metrics"]["estimator"]
    lines.extend(
        [
            "",
            f"Best position RMSE in this run: `{best}`.",
            "",
            "Generated files:",
            "- `metrics_summary.csv`: scalar metrics for each estimator.",
            "- `*_aligned_trajectory.csv`: aligned estimate vs truth at each matched timestamp.",
            "- `trajectory_planes.png`: XY, XZ, and YZ overlays.",
            "- `position_components.png`: x/y/z traces over time.",
            "- `position_error.png`: instantaneous Euclidean position error.",
            "- `orientation_error.png`: instantaneous orientation error.",
            "- `position_error_cdf.png`: cumulative distribution of position error.",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def make_plots(results: list[dict], output_dir: Path) -> None:
    colors = {
        "openvins": "#1f77b4",
        "orb": "#d62728",
        "vggt": "#2ca02c",
        "ground_truth": "#111111",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axis_pairs = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    for ax, (i, j, label) in zip(axes, axis_pairs):
        gt_positions = results[0]["gt_positions"]
        ax.plot(gt_positions[:, i], gt_positions[:, j], color=colors["ground_truth"], linewidth=2.5, label="ground truth")
        for result in results:
            name = result["metrics"]["estimator"]
            ax.plot(result["est_positions"][:, i], result["est_positions"][:, j], linewidth=1.5, color=colors.get(name, None), label=name)
        ax.set_title(f"{label} trajectory")
        ax.set_xlabel(["x (m)", "x (m)", "y (m)"][axis_pairs.index((i, j, label))])
        ax.set_ylabel(["y (m)", "z (m)", "z (m)"][axis_pairs.index((i, j, label))])
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_planes.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    component_labels = ["x", "y", "z"]
    for idx, ax in enumerate(axes):
        gt_time = results[0]["time_from_start"]
        gt_values = results[0]["gt_positions"][:, idx]
        ax.plot(gt_time, gt_values, color=colors["ground_truth"], linewidth=2.0, label="ground truth")
        for result in results:
            name = result["metrics"]["estimator"]
            ax.plot(result["time_from_start"], result["est_positions"][:, idx], linewidth=1.2, color=colors.get(name, None), label=name)
        ax.set_ylabel(f"{component_labels[idx]} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("time from estimator start (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "position_components.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for result in results:
        name = result["metrics"]["estimator"]
        ax.plot(result["time_from_start"], result["position_error"], linewidth=1.5, color=colors.get(name, None), label=name)
    ax.set_title("Position error over time")
    ax.set_xlabel("time from estimator start (s)")
    ax.set_ylabel("position error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "position_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for result in results:
        name = result["metrics"]["estimator"]
        ax.plot(
            result["time_from_start"],
            result["orientation_error_deg"],
            linewidth=1.5,
            color=colors.get(name, None),
            label=name,
        )
    ax.set_title("Orientation error over time")
    ax.set_xlabel("time from estimator start (s)")
    ax.set_ylabel("orientation error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        name = result["metrics"]["estimator"]
        errors = np.sort(result["position_error"])
        cdf = np.linspace(0.0, 1.0, len(errors), endpoint=True)
        ax.plot(errors, cdf, linewidth=1.8, color=colors.get(name, None), label=name)
    ax.set_title("Position error CDF")
    ax.set_xlabel("position error (m)")
    ax.set_ylabel("cumulative probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "position_error_cdf.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare estimator trajectories against ground truth.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("/workspaces/slam_project/comparison"),
        help="Directory containing the estimator bags and ground truth CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots and metrics. Defaults to <comparison-dir>/results.",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_dir.resolve()
    output_dir = (args.output_dir or (comparison_dir / "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth_csv(comparison_dir / "table_01.csv")
    orb_dir = comparison_dir / "t1_orb2"
    if not orb_dir.exists():
        orb_dir = comparison_dir / "t1_orb"

    estimators = {
        "openvins": comparison_dir / "t1_openvins",
        "orb": orb_dir,
        "vggt": comparison_dir / "t1_vggt",
    }

    results = []
    for name, bag_dir in estimators.items():
        trajectory = load_bag_trajectory(name, bag_dir)
        results.append(summarize_trajectory(trajectory, gt))

    write_metrics_csv(results, output_dir)
    write_aligned_trajectories(results, output_dir)
    write_report(results, output_dir)
    make_plots(results, output_dir)


if __name__ == "__main__":
    main()
