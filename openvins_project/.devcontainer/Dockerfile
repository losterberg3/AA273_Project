# Use the native ARM64 image for your Mac
FROM ros:humble-ros-base

# 1. Setup User (matches your Mac UID to avoid permission issues)
ARG USERNAME=nolan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 2. Install OpenVINS Dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    libeigen3-dev \
    libboost-all-dev \
    libceres-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
WORKDIR /home/workspace