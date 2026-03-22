# TNSRE-2025-01452
Tested during the peer review process.

## Version Requirements

- Create an anaconda environment via: `conda create -n BCI-DRL python=3.8`
- Activate the virtual env via: `conda activate BCI-DRL`
- Install the requirements via: `pip install -r requirements.txt`

## BCI

### Training

In the folder `BCI`, the script `ECCA.py` is used for offline training. Due to project limitations, only one subject dataset is provided in the `datasets` folder.

The `dataProcessor.py` file encapsulates the `EEGDataProcessor` class for processing EEG signals.

The `ReceiveData.py` file contains the `LSLDataCollector` class for real-time EEG data acquisition.

To collect EEG data in real-time, you must use an EEG acquisition device that supports the **Lab Streaming Layer (LSL)**. The installation package for LSL is located in the directory `\utils\labstreaminglayer-master`. The `ReceiveData` script, which is based on LSL, is responsible for acquiring EEG data. Please modify line 19 of this script to match the channel presets for your specific EEG device.

**Stimulation Interface**: Located in the folder `utils/SSVEP_App-master/bin/SSVEP_App_debug.exe`. To run this, you need to download and install [openFrameworks 0.10.1](https://openframeworks.cc/) and [Visual Studio 2017](https://visualstudio.microsoft.com/). 

## Simulation & Visualization 

Configuring the full CarSim environment for vehicle dynamics co-simulation can be highly cumbersome due to strict software dependencies, version mismatches, and licensing constraints. 

To ensure a smooth and accessible evaluation process for the reviewers, we have integrated a lightweight, standalone 2D visualization module using `matplotlib`. By enabling the `show_animation = True` flag in the environment scripts, you can directly run and visualize the online execution of the algorithms without needing CarSim. 

This module (`plot_car` and the animation block) plots the real-time vehicle kinematics, course tracking, safety boundaries, and target points:

```python
if show_animation:
    plt.cla()
    plt.plot(cx, cy, "-r", label="course")
    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    plot_car(state.x, state.y, state.yaw, steer=self.di)
    plt.plot(self.x_left, self.y_left)
    plt.plot(self.x_right, self.y_right)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Time[s]:" + str(round(time, 2))
              + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
    plt.pause(0.001)
```

## `BCI-FT`

The `env_cars` class is constructed in `MPC/BCI-FT.py`.

This file implements the `BCI-FT` algorithm described as Algorithm 1 in the corresponding paper.

## `BCI-DRL`

The `env_cars` class is used to enable environment interaction for all DRL methods.

`DQN.py` includes implementations of both `DDQN` and `DUE` algorithms.

`PPO.py` contains the implementation of the `PPO` algorithm.

`SAC.py` contains the implementation of the `SAC` algorithm.







