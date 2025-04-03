# 🦿 Legged Robotics in Genesis

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)

## Table of Contents

- [🦿 Legged Robotics in Genesis](#-legged-robotics-in-genesis)
  - [Table of Contents](#table-of-contents)
  - [🌟 Features](#-features)
  - [🧪 Test Results](#-test-results)
  - [🛠 Installation](#-installation)
  - [👋 Usage](#-usage)
    - [🚀 Quick Start](#-quick-start)
    - [📖 Instructions](#-instructions)
  - [🖼️ Gallery](#️-gallery)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [TODO](#todo)

---
## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**
  
  It's easy to use for those who are familiar with legged_gym and rsl_rl

- **Faster and Smaller**
  
  For a go2 walking on the plane task with 4096 envs, the training speed in Genesis is approximately **1.3x** compared to [Isaac Gym](https://developer.nvidia.com/isaac-gym), while the graphics memory usage is roughly **1/2** compared to IsaacGym.

  With this smaller memory usage, it's possible to **run more parallel environments**, which can further improve the training speed.

## 🧪 Test Results

For tests conducted on Genesis, please refer to [tests.md](./test_resources/tests.md)

## 🛠 Installation

1. Create a new python virtual env with python>=3.10
2.
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
   ```
3.
   ```bash
   pip install genesis-world=0.2.1
   ```
4. Install rsl_rl and tensorboard
   ```bash
   # Install rsl_rl.
   git clone https://github.com/leggedrobotics/rsl_rl
   cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517

   # Install tensorboard.
   pip install tensorboard
   ```
5. Install dependency
   ```bash
   git clone https://github.com/JingXunLin/Genesis_Go2.git
   cd Genesis_Go2
   pip install -e .
   ```

## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helper.py`), we can run a training session with the following command:

```bash
cd legged_gym/scripts
python train.py --headless # run training without rendering
```

After the training is done, paste the `run_name` under `logs/go2` to `load_run` in `go2_config.py`: 

![](./test_resources/paste_load_run.png)

Then, run `play.py` to visualize the trained model.

### 📖 Instructions

For more detailed instructions, please refer to the [wiki page](https://github.com/lupinjia/genesis_lr/wiki)

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
