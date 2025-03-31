# Isaac Imitation Learning
This repo provides the tools to capture demonstrations in IsaacLab, as well as train a robot using the collected demonstrations. Demonstrations can be collected using different means, for example keybaord, gamepad or a MetaQuest 3 headset. The imitation learning part focuses mainly on diffusion.

## Requirements
- Needed
    - Python 3.10
    - IsaacSim >= 4.5
    - IsaacLab >= 2.0
- Optional
    - [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
    - [SimPub](https://github.com/tsnz/SimPublisher)
    - [IRXR](https://github.com/tsnz/IRXR-Unity)

diffusion_policy is only needed when using the imitation learning part. SimPub is only needed when using the MQ3. IRXR is a seperate Unity application which needs to be installed on the MQ3.

## Installation

### 1. Create Conda environment using Python 3.10
```
conda create --name isaaclab python=3.10
conda activate isaaclab
```

### 2. Install IsaacLab and pytorch3d (for diffusion_policy if needed)
Example for Ubuntu 24.04 based on the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) with the addition of also installing pytorch3d
```
# torch
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch3d cuda-version=12.4

# alternatively, if diffusion_policy won't be installed, pip can be used as seen in the installation guide
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# IsaacSim
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Test if IsaacSim works
isaacsim

# Needed on Ubuntu
sudo apt install cmake build-essential

# IsaacLab
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
# this replaces our conda installed pytorch version with pip pytorch 2.5.1
# if the newly installed version differs from the old one this might have to be fixed


# Test if IsaacLab works
python scripts/tutorials/00_sim/create_empty.py
```

### 3. (Optional) Install diffusion_policy
```
cd ..
git clone git@github.com:real-stanford/diffusion_policy.git
cd diffusion_policy
pip install -e .
```

### 4. (Optional) Install SimPub
Only needed if MQ3 support for visualization and teleoperation is desired
```
cd ..
git clone git@github.com:tsnz/SimPublisher.git
cd SimPublisher
git checkout feat/isaacsim_deformable
pip install -e .
```

### 5. Install IsaacImitationLearning
```
cd ..
git clone git@github.com:tsnz/IsaacImitationLearning.git
cd IsaacImitationLearning
# depending on what should be included / checked
pip install -e .                                    # minimal installation
pip install -e .["SIMPUB", "IMITATION_LEARNING"]    # full installation
# or any other combination
```

### 6. (Optional) IRXR
Software used on MQ3 for teleoperation, for more information see IRXR repo
```
cd ..
git clone git@github.com:tsnz/IRXR-Unity.git
cd IRXR-Unity
git checkout feat/deformable-mq3
```

## Usage

### Setting up VS Code
Open the root folder of this repo in VS Code and execute the task "Setup VS Code". This will provide a similar setup to the setup task provided in IsaacLab

### Capturing demonstrations
The `record_demos.py` script can be used to record demonstration. It is generally better to first capture the demonstrations of a "LowDim" environment, since capturing demonstrations using cameras has a big impact on performance. The general workflow could be:
```
# capture 10 low dim demos
python record_demos.py --task [TASKNAME]-LowDim --num_demos 10 --teleop_device keyboard --dataset_file [OUTPUT_PATH]

# generate demos which can include cameras, since performance here is not relevant
python gen_obs_from_demos.py --task [TASKNAME] --dataset_file [OUTPUT_PATH] --output_file[FULL_DEMO_OUTPUT_PATH] --enable-cameras
```

This was the performance hit of using cameras does not negatively affect the operator when collecting demonstrations.
Capture sessions can be split up and the resulting files can be combined into a single file using `merge_hdf5_datasets.py`.

### Training a model
After collecting demonstrations, modify the corresponding config.yaml file in `cfg/tasks`. Only the `dataset_path` variable has to be changed if no other changes were made. After the change is made a corresponding workspace cfg has to be used for the task.
| CFG           | Task                  |
|:----------    |:----------            |
| lowdim        | lowdim                |
| hybrid        | lowdim, image, depth  |
| image         | lowdim, image, depth  |

```
# training a model
python train.py --config-name train_diffusion_unet_image_workspace task=lift_teddy_image_isaac
```
Results are logged to wandb, for which a login is required.

### Running model
There are two ways to run the model:
- Rerun rollout phase which is used during training
- Run in own env
```
# rerun rollout phase, --show is needed to disable IsaacLab headless mode
python eval.py --checkpoint [MODEL_CKPT_PATH] --out [RESULT_OUT] --show

# run in custom env
python play.py --checkpoint [MODEL_CKPT_PATH] --num_envs 4 --num_rollouts 2
```

The second option allows overwriting certain settings which are fixed when rerunning the rollout 
