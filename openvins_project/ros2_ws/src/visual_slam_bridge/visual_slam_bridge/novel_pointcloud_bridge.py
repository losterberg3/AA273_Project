from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class NovelPointCloudBridge(Node):
    def __init__(self) -> None:
        super().__init__("novel_pointcloud_bridge")

        self.declare_parameter("input_pose_topic", "/ov_msckf/poseimu")
        self.declare_parameter("output_pose_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("input_cloud_topic", "/ov_msckf/points_slam")
        self.declare_parameter("output_cloud_topic", "/visual_slam/mapping/point_cloud")
        self.declare_parameter("point_epsilon", 1.0e-4)
        self.declare_parameter("queue_size", 10)

        input_pose_topic = self.get_parameter("input_pose_topic").value
        output_pose_topic = self.get_parameter("output_pose_topic").value
        input_cloud_topic = self.get_parameter("input_cloud_topic").value
        output_cloud_topic = self.get_parameter("output_cloud_topic").value
        self._point_epsilon = float(self.get_parameter("point_epsilon").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self._seen_points = set()
        self._pose_pub = self.create_publisher(PoseStamped, output_pose_topic, queue_size)
        self._cloud_pub = self.create_publisher(PointCloud2, output_cloud_topic, queue_size)

        self.create_subscription(
            PoseWithCovarianceStamped,
            input_pose_topic,
            self._handle_pose,
            queue_size,
        )
        self.create_subscription(
            PointCloud2,
            input_cloud_topic,
            self._handle_cloud,
            queue_size,
        )

    def _handle_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self._pose_pub.publish(pose_msg)

    def _handle_cloud(self, msg: PointCloud2) -> None:
        novel_points = []

        for point in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            key = tuple(int(round(axis / self._point_epsilon)) for axis in point)
            if key in self._seen_points:
                continue
            self._seen_points.add(key)
            novel_points.append((float(point[0]), float(point[1]), float(point[2])))

        if not novel_points:
            return

        cloud_msg = point_cloud2.create_cloud_xyz32(msg.header, novel_points)
        self._cloud_pub.publish(cloud_msg)


def main() -> None:
    rclpy.init()
    node = NovelPointCloudBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
