---
license: apache-2.0
language:
  - en
  - zh
task_categories:
  - robotics
tags:
  - RoboCoin
  - LeRobot
size_categories: <1K
configs:
- config_name: default
  data_files: data/*/*.parquet
---

## Dataset Authors
This dataset is contributed by [[RoboCoin](https://RoboCoin.github.io)]

This dataset is annotated by [[RoboCoin](https://RoboCoin.github.io)]

## Dataset Description
This dataset uses an extended format based on [LeRobot](https://github.com/huggingface/lerobot) and is fully compatible with LeRobot.

- **Homepage:** https://RoboCoin.github.io/
- **Paper:** in comming
- **License:** apache-2.0

## Dataset Tags

- RoboCoin

- LeRobot


## Task Descriptions
### tasks
build blocks

### sub_tasks
Place the arch-shaped block in the center of view with the right gripper
Grasp the triangle-shaped block with the left gripper
Grasp the arch-shaped block with the right gripper
Grasp the triangle-shaped block with the right gripper
Place the box-shaped block onto the arch-shaped block with the right gripper
Grasp the arch-shaped block with the left gripper
Place the triangle-shaped block onto the box-shaped block with the right gripper
Grasp the box-shaped block with the left gripper
Place the arch-shaped block in the center of view with the left gripper
Place the triangle-shaped block onto the box-shaped block with the left gripper
Grasp the box-shaped block with the right gripper
Place the box-shaped block onto the arch-shaped block with the left gripper

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": "realman",
    "total_episodes": 498,
    "total_frames": 595002,
    "total_tasks": 1,
    "total_videos": 1494,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 25,
    "splits": {
        "train": "0:498"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "states": {
            "dtype": "float32",
            "shape": [
                30
            ],
            "name": [
                "right_arm_joint_1",
                "right_arm_joint_2",
                "right_arm_joint_3",
                "right_arm_joint_4",
                "right_arm_joint_5",
                "right_arm_joint_6",
                "right_arm_joint_7",
                "right_gripper_joint",
                "right_end_effector_positions_x",
                "right_end_effector_positions_y",
                "right_end_effector_positions_z",
                "right_end_effector_quat_x",
                "right_end_effector_quat_y",
                "right_end_effector_quat_z",
                "right_end_effector_quat_w",
                "left_arm_joint_1",
                "left_arm_joint_2",
                "left_arm_joint_3",
                "left_arm_joint_4",
                "left_arm_joint_5",
                "left_arm_joint_6",
                "left_arm_joint_7",
                "left_gripper_joint",
                "left_end_effector_positions_x",
                "left_end_effector_positions_y",
                "left_end_effector_positions_z",
                "left_end_effector_quat_x",
                "left_end_effector_quat_y",
                "left_end_effector_quat_z",
                "left_end_effector_quat_w"
            ]
        },
        "actions": {
            "dtype": "float32",
            "shape": [
                30
            ],
            "name": [
                "right_arm_joint_1",
                "right_arm_joint_2",
                "right_arm_joint_3",
                "right_arm_joint_4",
                "right_arm_joint_5",
                "right_arm_joint_6",
                "right_arm_joint_7",
                "right_gripper_joint",
                "right_end_effector_positions_x",
                "right_end_effector_positions_y",
                "right_end_effector_positions_z",
                "right_end_effector_quat_x",
                "right_end_effector_quat_y",
                "right_end_effector_quat_z",
                "right_end_effector_quat_w",
                "left_arm_joint_1",
                "left_arm_joint_2",
                "left_arm_joint_3",
                "left_arm_joint_4",
                "left_arm_joint_5",
                "left_arm_joint_6",
                "left_arm_joint_7",
                "left_gripper_joint",
                "left_end_effector_positions_x",
                "left_end_effector_positions_y",
                "left_end_effector_positions_z",
                "left_end_effector_quat_x",
                "left_end_effector_quat_y",
                "left_end_effector_quat_z",
                "left_end_effector_quat_w"
            ]
        },
        "next.done": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```

## Citation
```bibtex

```