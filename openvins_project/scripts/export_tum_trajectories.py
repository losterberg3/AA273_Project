#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def read_metadata_topic(metadata_path: Path) -> tuple[str, str]:
    metadata = yaml.safe_load(metadata_path.read_text())
    topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    for entry in topics:
        topic = entry["topic_metadata"]["name"]
        msg_type = entry["topic_metadata"]["type"]
        if msg_type in {"nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped"}:
            return topic, msg_type
    raise RuntimeError(f"No supported odometry topic found in {metadata_path}")


def export_ground_truth_csv_to_tum(csv_path: Path, tum_path: Path) -> None:
    with csv_path.open("r", newline="") as src, tum_path.open("w") as dst:
        reader = csv.DictReader(src)
        for row in reader:
            timestamp = int(row["#time(ns)"]) * 1e-9
            dst.write(
                f"{timestamp:.9f} "
                f"{float(row['px']):.9f} {float(row['py']):.9f} {float(row['pz']):.9f} "
                f"{float(row['qx']):.9f} {float(row['qy']):.9f} {float(row['qz']):.9f} {float(row['qw']):.9f}\n"
            )


def export_bag_to_tum(bag_dir: Path, tum_path: Path) -> None:
    topic_name, msg_type_name = read_metadata_topic(bag_dir / "metadata.yaml")
    msg_type = get_message(msg_type_name)

    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_dir), storage_id="sqlite3"), ConverterOptions("", ""))

    with tum_path.open("w") as dst:
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != topic_name:
                continue

            msg = deserialize_message(data, msg_type)
            stamp = msg.header.stamp
            pose = msg.pose.pose if msg_type_name == "nav_msgs/msg/Odometry" else msg.pose
            timestamp = stamp.sec + stamp.nanosec * 1e-9
            dst.write(
                f"{timestamp:.9f} "
                f"{pose.position.x:.9f} {pose.position.y:.9f} {pose.position.z:.9f} "
                f"{pose.orientation.x:.9f} {pose.orientation.y:.9f} {pose.orientation.z:.9f} {pose.orientation.w:.9f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export slam_project trajectories to TUM format.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("/workspaces/slam_project/comparison"),
        help="Directory containing estimator bag folders and table_01.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for TUM files. Defaults to <comparison-dir>/evo.",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_dir.resolve()
    output_dir = (args.output_dir or comparison_dir / "evo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    orb_dir = comparison_dir / "t1_orb2"
    if not orb_dir.exists():
        orb_dir = comparison_dir / "t1_orb"

    export_ground_truth_csv_to_tum(comparison_dir / "table_01.csv", output_dir / "table_01.txt")
    export_bag_to_tum(comparison_dir / "t1_openvins", output_dir / "openvins_CameraTrajectory.txt")
    export_bag_to_tum(orb_dir, output_dir / "orb_CameraTrajectory.txt")
    export_bag_to_tum(comparison_dir / "t1_vggt", output_dir / "vggt_CameraTrajectory.txt")


if __name__ == "__main__":
    main()
