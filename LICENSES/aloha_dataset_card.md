---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

This dataset is a lerobot conversion of the `aloha_pen_uncap_diverse` subset of BiPlay.

BiPlay contains 9.7 hours of bimanual data collected with an aloha robot at the RAIL lab @ UC Berkeley, USA. It contains 7023 clips, 2000 language annotations and 326 unique scenes.

Paper: https://huggingface.co/papers/2410.10088 Code: https://github.com/sudeepdasari/dit-policy If you use the dataset please cite:

```
@inproceedings{dasari2025ingredients,
  title={The Ingredients for Robotic Diffusion Transformers},
  author={Sudeep Dasari and Oier Mees and Sebastian Zhao and Mohan Kumar Srirama and Sergey Levine},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  address = {Atlanta, USA}
}
```

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "aloha",
    "total_episodes": 123,
    "total_frames": 36900,
    "total_tasks": 1,
    "total_videos": 0,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 50,
    "splits": {
        "train": "0:123"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": [
                [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper"
                ]
            ]
        },
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_low": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": [
                3,
                480,
                640
            ],
            "names": [
                "channels",
                "height",
                "width"
            ]
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

**BibTeX:**

```bibtex
[More Information Needed]
```