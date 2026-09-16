---
license: cc-by-4.0
task_categories:
- robotics
tags:
- robotics
- manipulation
- libero
- vision-language-action
- multi-task
- converted
- lerobot
pretty_name: LIBERO Converted
size_categories:
- 100K<n<1M
---

# LIBERO Dataset for LeRobot

This is the [LIBERO dataset](https://huggingface.co/datasets/physical-intelligence/libero), **preprocessed by the HuggingFace VLA team to fit the LeRobot framework**.  

The dataset is structured for easy use in VLA (Vision-Language-Action) LeRobot training pipelines, so you can easily train and evaluate models without extra preprocessing.

## Dataset Description

LIBERO (Language-Integrated Benchmark for Evaluating Robot Manipulation) is a comprehensive benchmark for robot manipulation tasks with natural language instructions. This dataset combines four individual Libero datasets: **Libero-Spatial**, **Libero-Object**, **Libero-Goal** and **Libero-10**. All datasets were taken from the original sources and converted into LeRobot format.

- **Homepage**: https://libero-project.github.io
- **Paper**: https://arxiv.org/abs/2306.03310
- **License**: CC-BY 4.0

This converted version maintains all original functionality while providing standardized feature naming conventions.

### Details

- **Episodes**: 1,693 manipulation episodes
- **Tasks**: 40 unique manipulation tasks
- **Robot**: Franka Panda 7-DOF arm
- **Cameras**: Dual camera setup (main view + wrist camera)
- **Resolution**: 256×256 RGB images
- **Action Space**: 7-dimensional (6 joint velocities + gripper)
- **FPS**: 10 Hz

## Dataset Structure

### Features

| Feature | Type | Shape | Description |
|---------|------|-------|-------------|
| `observation.images.image` | Image | (256, 256, 3) | Main camera RGB view |
| `observation.images.image2` | Image | (256, 256, 3) | Wrist camera RGB view |
| `observation.state` | Tensor | (8,) | Joint positions + gripper state |
| `action` | Tensor | (7,) | Target joint velocities + gripper action |
| `timestamp` | Float | () | Episode timestamp |
| `frame_index` | Int | () | Frame index within episode |
| `episode_index` | Int | () | Episode identifier |
| `index` | Int | () | Global frame index |
| `task_index` | Int | () | Task identifier (0-39) |

### Task Categories

The dataset includes 40 diverse manipulation tasks across multiple categories:
- **Pick and Place**: Object manipulation and placement
- **Drawer Operations**: Opening and closing drawers
- **Door Operations**: Handle manipulation and door opening/closing
- **Assembly Tasks**: Multi-object coordination and assembly
- **Tool Use**: Utilizing tools for manipulation tasks

## Usage

### Loading the Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load the converted dataset
dataset = LeRobotDataset("HuggingFaceVLA/libero")

# Access a sample
sample = dataset[0]
main_image = sample["observation.images.image"]
wrist_image = sample["observation.images.image2"] 
robot_state = sample["observation.state"]
action = sample["action"]
```

## Conversion Details

This dataset was converted from the original LIBERO v2.0 format to LeRobot v2.1 format with the following improvements:

1. **Standardized Feature Names**: Renamed keys to follow LeRobot conventions
2. **Added Episode Statistics**: Generated `episodes_stats.jsonl` for efficient loading
3. **Updated Metadata**: Compatible with latest LeRobot framework
4. **Structured Image Keys**: Organized image features under `observation.images.*`
5. **Preserved All Data**: No data loss during conversion

### Original to Converted Key Mapping

| Original Key | Converted Key |
|--------------|---------------|
| `image` | `observation.images.image` |
| `wrist_image` | `observation.images.image2` |
| `state` | `observation.state` |
| `actions` | `action` |

## Dataset Statistics

- **Total Episodes**: 1,693
- **Total Frames**: 273,465
- **Average Episode Length**: 161.6 frames
- **Dataset Size**: ~35GB
- **Format**: LeRobot v2.1

## Citation

If you use this dataset, please cite the original LIBERO paper:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zeng, Yifeng and Patil, Arjun and Mu, Xingyu and Xu, Qingqing and Liu, Ying and Wu, Stone and Liu, Yurong and Tenenbaum, Joshua B and others},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

## License

This dataset is released under the CC-BY 4.0 License, following the original LIBERO dataset licensing.

## Original Dataset

- **Original Source**: [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)
- **Paper**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- **Homepage**: [https://libero-project.github.io](https://libero-project.github.io)

## Conversion Process

This dataset was converted using the LeRobot conversion pipeline:

1. Downloaded original dataset from HuggingFace Hub
2. Applied LeRobot v2.0 to v2.1 conversion
3. Renamed features to standard conventions
4. Generated episode statistics
5. Validated data integrity
6. Uploaded with proper metadata

For questions about the conversion or issues with the dataset, please open an issue on the [LeRobot repository](https://github.com/huggingface/lerobot).
