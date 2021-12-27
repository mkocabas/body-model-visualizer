import open3d as o3d
import torch
from smplx import SMPL, SMPLH, SMPLX, MANO, FLAME

betas = torch.zeros(1, 10)
global_orient = torch.zeros(1, 3)

NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS = 15
NUM_FACE_JOINTS = 3

for bm in SMPL, SMPLX, MANO, FLAME:

    extra_params = {}
    if bm.__name__ in ('SMPLX', 'MANO'):
        extra_params['use_pca'] = False
        extra_params['use_face_contour'] = True
    model = bm(f'data/body_models/{bm.__name__}', **extra_params)
    input_args = {}
    if bm.__name__ == 'SMPL':
        input_args = {
            'body_pose': torch.zeros(1, model.NUM_BODY_JOINTS * 3)
        }
    elif bm.__name__ == 'SMPLX':
        input_args = {
            'body_pose': torch.zeros(1, model.NUM_BODY_JOINTS * 3),
            'left_hand_pose': torch.zeros(1, model.NUM_HAND_JOINTS * 3),
            'right_hand_pose': torch.zeros(1, model.NUM_HAND_JOINTS * 3),
            'jaw_pose': torch.zeros(1, 3),
            'leye_pose': torch.zeros(1, 3),
            'reye_pose': torch.zeros(1, 3),
        }
    elif bm.__name__ == 'MANO':
        input_args = {
            'hand_pose': torch.zeros(1, model.NUM_HAND_JOINTS * 3)
        }
    elif bm.__name__ == 'FLAME':
        input_args = {
            'expression': torch.zeros(1, 10),
            'jaw_pose': torch.zeros(1, 3),
            'neck_pose': torch.zeros(1, 3),
            'leye_pose': torch.zeros(1, 3),
            'reye_pose': torch.zeros(1, 3),
        }

    model_output = model(global_orient=global_orient, betas=betas, **input_args)
    print(f'{bm.__name__} - NUM_BODY_JOINTS {model.NUM_BODY_JOINTS}, NUM_JOINTS {model.NUM_JOINTS}, NUM_BETAS {model.num_betas}')
    for k,v in model_output.items():
        if isinstance(v, torch.Tensor):
            print(f'{bm.__name__}-{k}: {v.shape}')

# model = SMPLX('data/body_models/smplx', gender='female')
# print(f'SMPLX - NUM_BODY_JOINTS {model.NUM_BODY_JOINTS}, NUM_JOINTS {model.NUM_JOINTS}, NUM_BETAS {model.num_betas}')
#
# model = MANO('data/body_models/mano')
# print(f'MANO - NUM_BODY_JOINTS {model.NUM_BODY_JOINTS}, NUM_JOINTS {model.NUM_JOINTS}, NUM_BETAS {model.num_betas}')
#
# model = FLAME('data/body_models/flame')
# print(f'FLAME - NUM_BODY_JOINTS {model.NUM_BODY_JOINTS}, NUM_JOINTS {model.NUM_JOINTS}, NUM_BETAS {model.num_betas}')
