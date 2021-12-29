# Body Model Visualizer

### Introduction

This is a simple Open3D-based GUI for SMPL-family body models. This GUI lets you
play with the shape, expression, and pose parameters of SMPL, SMPL-X, MANO, FLAME
body models. Features include:

- Interactive editing of shape, expression, pose parameters


https://user-images.githubusercontent.com/6137870/147476574-983063a8-233b-400c-bd64-7d946578919b.mp4


- Visualize body model joints and joint names


https://user-images.githubusercontent.com/6137870/147476577-39cd3a59-1add-4e2d-8c87-406ef964b558.mp4


- Simple IK solver to match an input pose


https://user-images.githubusercontent.com/6137870/147476585-9bbc0018-9220-4efa-9f4f-f37fdcf35db9.mp4


- Save edited model parameters


https://user-images.githubusercontent.com/6137870/147476590-d1b3e275-207e-4b30-99d6-0386f5ab74c5.mp4


- View controls


https://user-images.githubusercontent.com/6137870/147476594-cf244338-c841-4f17-a221-98038fdd9f4a.mp4


- Lighting controls


https://user-images.githubusercontent.com/6137870/147476612-ccd73006-4e7d-4caf-ae99-50418444f1fa.mp4


- Material settings


https://user-images.githubusercontent.com/6137870/147476625-8d019582-2a15-41f8-ae7f-93435e7e2529.mp4


- Web visualization support

Even though there are existing Blender/Unity plugins for these models, our main
audience here is researchers who would like to quickly edit/visualize body models
without the need to install a graphics software.


## Installation

Clone the repo and install the requirements (use python3.9).

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
If you want to enable web visualization, run:
```shell
python main.py --web
```

## Guidelines

### Saved model parameters
`File > Save Model Params` lets you save the edited body model parameters. Output is a pickled
python dictionary with below keys:
```shell
dict_keys(['betas', 'expression', 'gender', 'body_model', 
           'joints', 'body_pose', 'global_orient'])
```
