# joint names are borrowed from Vassilis' ExPose code
# https://github.com/vchoutas/expose/blob/master/expose/data/targets/keypoints.py

import numpy as np
import open3d as o3d


smpl_joint_names = [
    # 'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'righ_index1',
]


smplx_body_joint_names = [
    # 'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]


hand_joint_names = [
    # 'wrist',        # 0
    'index1',       # 1
    'index2',       # 2
    'index3',       # 3
    'middle1',      # 4
    'middle2',      # 5
    'middle3',      # 6
    'pinky1',       # 7
    'pinky2',       # 8
    'pinky3',       # 9
    'ring1',        # 10
    'ring2',        # 11
    'ring3',        # 12
    'thumb1',       # 13
    'thumb2',       # 14
    'thumb3',       # 15
]


SMPLX_PARTS = {
    'pelvis': 'body',
    'left_hip': 'body',
    'right_hip': 'body',
    'spine1': 'body',
    'left_knee': 'body',
    'right_knee': 'body',
    'spine2': 'body',
    'left_ankle': 'body',
    'right_ankle': 'body',
    'spine3': 'body',
    'left_foot': 'body',
    'right_foot': 'body',
    'neck': 'body,flame',
    'left_collar': 'body',
    'right_collar': 'body',
    'head': 'body,head,flame',
    'left_shoulder': 'body',
    'right_shoulder': 'body',
    'left_elbow': 'body',
    'right_elbow': 'body',
    'left_wrist': 'body',
    'right_wrist': 'body',
    'jaw': 'body,head,flame',
    'left_eye_smplx': 'body,head,flame',
    'right_eye_smplx': 'body,head,flame',
    'left_index1': 'hand',
    'left_index2': 'hand',
    'left_index3': 'hand',
    'left_middle1': 'hand',
    'left_middle2': 'hand',
    'left_middle3': 'hand',
    'left_pinky1': 'hand',
    'left_pinky2': 'hand',
    'left_pinky3': 'hand',
    'left_ring1': 'hand',
    'left_ring2': 'hand',
    'left_ring3': 'hand',
    'left_thumb1': 'hand',
    'left_thumb2': 'hand',
    'left_thumb3': 'hand',
    'right_index1': 'hand',
    'right_index2': 'hand',
    'right_index3': 'hand',
    'right_middle1': 'hand',
    'right_middle2': 'hand',
    'right_middle3': 'hand',
    'right_pinky1': 'hand',
    'right_pinky2': 'hand',
    'right_pinky3': 'hand',
    'right_ring1': 'hand',
    'right_ring2': 'hand',
    'right_ring3': 'hand',
    'right_thumb1': 'hand',
    'right_thumb2': 'hand',
    'right_thumb3': 'hand',
    'nose': 'body,head',
    'right_eye': 'body,head',
    'left_eye': 'body,head',
    'right_ear': 'body,head',
    'left_ear': 'body,head',
    'left_big_toe': 'foot',
    'left_small_toe': 'foot',
    'left_heel': 'foot',
    'right_big_toe': 'foot',
    'right_small_toe': 'foot',
    'right_heel': 'foot',
    'left_thumb': 'hand',
    'left_index': 'hand',
    'left_middle': 'hand',
    'left_ring': 'hand',
    'left_pinky': 'hand',
    'right_thumb': 'hand',
    'right_index': 'hand',
    'right_middle': 'hand',
    'right_ring': 'hand',
    'right_pinky': 'hand',
    'right_eye_brow1': 'face,head,flame',
    'right_eye_brow2': 'face,head,flame',
    'right_eye_brow3': 'face,head,flame',
    'right_eye_brow4': 'face,head,flame',
    'right_eye_brow5': 'face,head,flame',
    'left_eye_brow5': 'face,head,flame',
    'left_eye_brow4': 'face,head,flame',
    'left_eye_brow3': 'face,head,flame',
    'left_eye_brow2': 'face,head,flame',
    'left_eye_brow1': 'face,head,flame',
    'nose1': 'face,head,flame',
    'nose2': 'face,head,flame',
    'nose3': 'face,head,flame',
    'nose4': 'face,head,flame',
    'right_nose_2': 'face,head,flame',
    'right_nose_1': 'face,head,flame',
    'nose_middle': 'face,head,flame',
    'left_nose_1': 'face,head,flame',
    'left_nose_2': 'face,head,flame',
    'right_eye1': 'face,head,flame',
    'right_eye2': 'face,head,flame',
    'right_eye3': 'face,head,flame',
    'right_eye4': 'face,head,flame',
    'right_eye5': 'face,head,flame',
    'right_eye6': 'face,head,flame',
    'left_eye4': 'face,head,flame',
    'left_eye3': 'face,head,flame',
    'left_eye2': 'face,head,flame',
    'left_eye1': 'face,head,flame',
    'left_eye6': 'face,head,flame',
    'left_eye5': 'face,head,flame',
    'right_mouth_1': 'face,head,flame',
    'right_mouth_2': 'face,head,flame',
    'right_mouth_3': 'face,head,flame',
    'mouth_top': 'face,head,flame',
    'left_mouth_3': 'face,head,flame',
    'left_mouth_2': 'face,head,flame',
    'left_mouth_1': 'face,head,flame',
    'left_mouth_5': 'face,head,flame',
    'left_mouth_4': 'face,head,flame',
    'mouth_bottom': 'face,head,flame',
    'right_mouth_4': 'face,head,flame',
    'right_mouth_5': 'face,head,flame',
    'right_lip_1': 'face,head,flame',
    'right_lip_2': 'face,head,flame',
    'lip_top': 'face,head,flame',
    'left_lip_2': 'face,head,flame',
    'left_lip_1': 'face,head,flame',
    'left_lip_3': 'face,head,flame',
    'lip_bottom': 'face,head,flame',
    'right_lip_3': 'face,head,flame',
    'right_contour_1': 'face,head,flame',
    'right_contour_2': 'face,head,flame',
    'right_contour_3': 'face,head,flame',
    'right_contour_4': 'face,head,flame',
    'right_contour_5': 'face,head,flame',
    'right_contour_6': 'face,head,flame',
    'right_contour_7': 'face,head,flame',
    'right_contour_8': 'face,head,flame',
    'contour_middle': 'face,head,flame',
    'left_contour_8': 'face,head,flame',
    'left_contour_7': 'face,head,flame',
    'left_contour_6': 'face,head,flame',
    'left_contour_5': 'face,head,flame',
    'left_contour_4': 'face,head,flame',
    'left_contour_3': 'face,head,flame',
    'left_contour_2': 'face,head,flame',
    'left_contour_1': 'face,head,flame'
}


SMPLX_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplx',
    'right_eye_smplx',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]


MANO_NAMES = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    # 'right_thumb',
    # 'right_index',
    # 'right_middle',
    # 'right_ring',
    # 'right_pinky'
]


SMPL_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'right_index1',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]


def get_part_idxs():
    body_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'body' in val])

    hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'hand' in val])

    left_hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'hand' in val and 'left' in SMPLX_NAMES[idx]])

    right_hand_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'hand' in val and 'right' in SMPLX_NAMES[idx]])

    face_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'face' in val])
    head_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'head' in val])
    flame_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'flame' in val])
    foot_idxs = np.asarray([
        idx
        for idx, val in enumerate(SMPLX_PARTS.values())
        if 'foot' in val])
    #  joint_weights[hand_idxs] = hand_weight
    #  joint_weights[face_idxs] = face_weight
    return {
        'body': body_idxs.astype(np.int64),
        'hand': hand_idxs.astype(np.int64),
        'face': face_idxs.astype(np.int64),
        'head': head_idxs.astype(np.int64),
        'left_hand': left_hand_idxs.astype(np.int64),
        'right_hand': right_hand_idxs.astype(np.int64),
        'flame': flame_idxs.astype(np.int64),
        'foot': foot_idxs.astype(np.int64),
    }


PARTS = get_part_idxs()
BODY_IDXS = PARTS['body']
LEFT_HAND_IDXS = PARTS['left_hand']
RIGHT_HAND_IDXS = PARTS['right_hand']
FACE_IDXS = PARTS['face']
FLAME_IDXS = PARTS['flame']
HEAD_IDXS = PARTS['head']
FOOT_IDXS = PARTS['foot']
LEFT_HAND_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in LEFT_HAND_IDXS]
RIGHT_HAND_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in RIGHT_HAND_IDXS]
HEAD_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in HEAD_IDXS]
FLAME_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in FLAME_IDXS]
FOOT_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in FOOT_IDXS]
BODY_KEYPOINT_NAMES = [SMPLX_NAMES[ii] for ii in BODY_IDXS]


def get_checkerboard_plane(plane_width=20, num_boxes=15, center=True):

    pw = plane_width/num_boxes
    # white = [0.8, 0.8, 0.8]
    # black = [0.2, 0.2, 0.2]
    white = [230./255., 244./255., 244./255.]
    black = [int(150 / 1.3)/255., int(217 / 1.3)/255., int(217 / 1.3)/255.]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            # ground = trimesh.primitives.Box(
            #     center=[0, 0, -0.0001],
            #     extents=[pw, pw, 0.0002]
            # )
            ground = o3d.geometry.TriangleMesh.create_box(width=pw, height=0.0002, depth=pw)

            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.translate([c[0], 0, c[1]])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.paint_uniform_color(black if ((i+j) % 2) == 0 else white)
            meshes.append(ground)

    return meshes


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()