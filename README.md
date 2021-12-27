# Body Model Visualizer

### Introduction

This is a simple Open3D-based GUI for SMPL-family body models. This GUI lets you
play with the shape, expression, and pose parameters of SMPL, SMPL-X, MANO, FLAME
body models. Features include:

- Interactive editing of shape, expression, pose parameters

https://user-images.githubusercontent.com/6137870/147475119-03de271f-115e-4ecf-816f-5c182a05dc12.mp4

- Visualize body model joints and joint names
- Simple IK solver to match an input pose
- Save edited model parameters
- View controls
- Lighting controls
- Material settings

Even though there are existing Blender/Unity plugins for these models, our main
audience here is researchers who would like to quickly edit/visualize body models
without the need to install a graphics software.


## Installation

Clone the repo and install the requirements. Note that we tested with Python 3.9.

```shell
pip install -r requirements.txt
```

Download the SMPL, SMPL-X, MANO, FLAME body models:

- SMPL: https://smpl.is.tue.mpg.de/ (v1.1.0)
- SMPL-X: https://smpl-x.is.tue.mpg.de/ (v1.1)
- MANO: https://mano.is.tue.mpg.de/
- FLAME: https://flame.is.tue.mpg.de/
  - For landmarks: https://github.com/soubhiksanyal/RingNet/blob/master/flame_model/

Copy downloaded files under `data/body_models`, this folder should look like:

```shell
data
└── body_models
    ├── flame
    │   ├── FLAME_FEMALE.pkl
    │   ├── FLAME_MALE.pkl
    │   ├── FLAME_NEUTRAL.pkl
    │   ├── flame_dynamic_embedding.npy
    │   └── flame_static_embedding.pkl
    ├── mano
    │   ├── MANO_LEFT.pkl
    │   └── MANO_RIGHT.pkl
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    └── smplx
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_MALE.npz
        └── SMPLX_NEUTRAL.npz

```

Finally, run:
```shell
python main.py
```

## Guidelines

### Saved model parameters
`File > Save Model Params` lets you save the edited body model parameters. Output is a pickled
python dictionary with below keys:
```shell
dict_keys(['betas', 'expression', 'gender', 'body_model', 
           'joints', 'body_pose', 'global_orient'])
```
