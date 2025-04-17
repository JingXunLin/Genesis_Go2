# 🦿 Legged Robotics in Genesis

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)


## 🛠 Installation

1. Create a new python virtual env with python>=3.10
2. Install pytorch for rocm
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
   ```
3. Must use pip install; use newest Genesis will cause some error
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

   pip install open3d
   pip install pygame
   ```
6. Fix OpenGL error
   ```bash
   conda install -c conda-forge libstdcxx-ng

   Modify  ~/miniconda3/envs/genesis/lib/python3.12/site-packages/genesis/ext/pyrender/constants.py
   TARGET_OPEN_GL_MAJOR = 3  # Target OpenGL Major Version
   TARGET_OPEN_GL_MINOR = 3  # Target OpenGL Minor Version
   ```
## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helper.py`), we can run a training session with the following command:

```bash
cd legged_gym/scripts
python train.py --headless --max_iterations=1000 # run training without rendering
```

After the training is done, paste the `run_name` under `logs/go2` to `load_run` in `go2_config.py`: 

Then, run `play.py` to visualize the trained model.
```bash
vglrun python play.py --task=go2_rough
```
### 📖 Instructions

For more detailed instructions, please refer to the [wiki page](https://github.com/lupinjia/genesis_lr/wiki)

## DEMO
- [result](https://youtu.be/wIMQZ1X2RRA)

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- [genesis_lr](https://github.com/lupinjia/genesis_lr)
