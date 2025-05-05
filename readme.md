# Isaac Imitation Learning
This repo provides the tools to capture demonstrations in IsaacLab, as well as train a robot using the collected demonstrations. Demonstrations can be collected using different means, for example keyboard, gamepad or a MetaQuest 3 headset. The imitation learning part focuses mainly on diffusion.

## Requirements

This repo uses [Git LFS](https://git-lfs.com/) to store usd assets and the corresponding textures.

- Needed
    - Python 3.10
    - IsaacSim >= 4.5
    - IsaacLab >= 2.0
- Optional
    - [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
    - [SimPub](https://github.com/tsnz/SimPublisher)
    - [IRXR](https://github.com/tsnz/IRXR-Unity)

diffusion_policy is only needed when using the imitation learning part. SimPub is only needed when using the MQ3. IRXR is a separate Unity application which needs to be installed on the MQ3 and communicates with the simulation using SimPub.

## Installation

### 1. Create Conda environment using Python 3.10
```
conda create --name isaaclab python=3.10
conda activate isaaclab
```

### 2. Install IsaacSim and IsaacLab
Example for Ubuntu 24.04 based on the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html):
```
# torch
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
./isaaclab.sh --install none


# Test if IsaacLab works
python scripts/tutorials/00_sim/create_empty.py
```

### 3. (Optional) Install diffusion_policy
```
cd ..
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
git clone git@github.com:real-stanford/diffusion_policy.git
cd diffusion_policy
pip install -e .
```

### 4. (Optional) Install SimPub
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
# or any combination
```

### 6. (Optional) IRXR
Software that needs to be installed on the MQ3 for teleoperation and visualization. For more information see IRXR repo.
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

This way the performance hit of using cameras does not negatively affect the operator when collecting demonstrations.
Capture sessions can be split up and the resulting files can be combined into a single file using `merge_hdf5_datasets.py`.

### Training a model
After collecting demonstrations, modify the corresponding config.yaml file in `cfg/tasks`. Only the `dataset_path` variable has to be changed if no other changes were made. After the change is made a corresponding workspace cfg has to be used for the task.
| Workspace CFG     | Supported Tasks       |
|:----------        |:----------            |
| lowdim            | lowdim                |
| image             | lowdim, image,        |
| hybrid_image      | lowdim, image         |
| mixed             | lowdim, image, depth  |
| hybrid_mixed      | lowdim, image, depth  |


hybrid_image and image use the original diffusion_policy workspaces and policies which do not support depth information as inputs. hybrid_image uses the observation encoder that comes with robomimic, image uses it's own observation encoder which can be specified in the configuration file. Mixed uses a custom observation encoder that supports depth inputs. This way a original diffusion_policy policy can be used. Mixed_hybrid is a custom version of the diffusion_policy hybrid policy which adds depth as a valid input modality. When a task uses depth information a custom dataset loader has to be used. The one that comes with diffusion_policy does not load depth information. This custom loader is compatible with diffusion_policy policies and workspaces if depth is not used as an input.

Be careful when enabling caching for the dataset loaders. Using different versions of the loaders does not automatically regenerate existing caches.


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

The second option allows overwriting certain settings which are fixed when rerunning the rollout.

### Docker
For information about using Docker or other containers see [Docker](docker/readme.md)
