from setuptools import setup

package_name = "visual_slam_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nolan Topper",
    maintainer_email="nolantopper@example.com",
    description="Relay OpenVINS outputs into visual_slam topics for bag recording.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "novel_pointcloud_bridge = visual_slam_bridge.novel_pointcloud_bridge:main",
        ],
    },
)
