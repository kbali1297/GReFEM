import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import pandas as pd
import argparse
from generate_views import read_pos_file
from scipy.spatial.transform import Rotation as R

def look_at_quaternion(cam_pos, target, up=np.array([0, 1, 0])):
    # Forward vector (camera -> target)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # Right vector
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # Recompute orthogonal up vector
    true_up = np.cross(forward, right)
    true_up /= np.linalg.norm(true_up)

    # Construct rotation matrix (world to camera)
    R_wc = np.stack([right, true_up, forward], axis=1)  # columns = camera axes
    rot = R.from_matrix(R_wc)
    quat = rot.as_quat()  # [x, y, z, w]
    return quat

def extract_camera_pose(pose_matrix, return_orientation=False):
    """Extract camera position and orientation as quaternion from pose matrix."""
    position = pose_matrix[:3, 3]
    orientation = R.from_matrix(pose_matrix[:3, :3])
    quaternion = orientation.as_quat()
    if return_orientation:
        return position, orientation
    return position, quaternion


def camera_distance(pose1, pose2, position_weight=1.0, orientation_weight=1.0):
    """
    Calculate a distance metric between two camera poses.
    Consider position and orientation (using quaternions).
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2

    # Position distance
    pos_distance = 0 #ignoring for now since all camera positions are equidistant from the actual object #np.linalg.norm(pos1 - pos2)

    # Orientation distance (angle between quaternions)
    quat_dot_product = np.abs(np.dot(quat1, quat2))
    orientation_distance = 2 * np.arccos(np.clip(quat_dot_product, -1.0, 1.0))

    # Combine distances with weights
    return (position_weight * pos_distance +
            orientation_weight * orientation_distance)


def calculate_image_distance_(image_poses):
    # Extract camera poses
    poses = {img: extract_camera_pose(pose) for img, pose in image_poses.items()}

    # Create a list of images
    image_names = list(poses.keys())

    # Calculate pairwise distances
    n = len(image_names)
    distance_df = pd.DataFrame(np.zeros((n, n)), index=image_names, columns=image_names)
    for i in range(n):
        for j in range(i + 1, n):
            dist = camera_distance(poses[image_names[i]], poses[image_names[j]])
            distance_df.iloc[i, j] = dist
            distance_df.iloc[j, i] = dist
    return distance_df


def load_pose(filename):
    lines = open(filename).read().splitlines()
    print(filename)
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return np.asarray(lines).astype(np.float32)


def calculate_view_distance(scene_id, args):
    pose_path = os.path.join(args.image_folder, scene_id, 'pose')
    pose_file_list = [i for i in os.listdir(pose_path) if i[-4:] == '.txt']
    pose_dict = {}
    for pose_file in pose_file_list:
        pose_file_ = os.path.join(pose_path, pose_file)
        pose_info = load_pose(pose_file_)
        image_name = '{}.jpg'.format(pose_file[:-4])
        pose_dict[image_name] = pose_info
    distance_df = calculate_image_distance_(pose_dict)
    return distance_df


def save_view_distance(args):
    scene_list = [i for i in os.listdir(args.image_folder) if i[:5] == 'scene']
    for scene_id in tqdm(scene_list):
        distance_df = calculate_view_distance(scene_id, args)
        distance_df.to_csv(os.path.join(args.view_distance_folder, '{}.csv'.format(scene_id)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")
    args = parser.parse_args()

    # args = load_and_update(args)
    # save_view_distance(args)

    main_geom_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries'

    for cad_folder in tqdm(os.listdir(main_geom_dir)):
        folder_path = os.path.join(main_geom_dir, cad_folder)
        POS_PATH = f"{folder_path}/{cad_folder}_grad_0.pos"
        zoom = 4.5
        cam_positions_and_quats = []
        
        top_views_path = f'{folder_path}/{cad_folder}_top_views.txt'
        sorted_views = []
        if os.path.exists(top_views_path):
            coords, vals = read_pos_file(POS_PATH)
            object_center = coords.mean(axis=0)
            radius = np.linalg.norm(coords - object_center, axis=1).max() * zoom

            with open(f'{folder_path}/{cad_folder}_top_views.txt', 'r') as fread:
                for line in fread:
                    sorted_views.append(line[:-1])
                    
                    azimuth = float(line[:-1].split(',')[0].split('View: ')[1].split('_a')[1])
                    elevation = float(line[:-1].split(',')[0].split('View: ')[1].split('_e')[1].split('_')[0])
                    alpha = np.pi/2 - np.radians(azimuth) # angle from z axis towards +x axis in the azimuthal plane y=0
                    beta =  np.pi/2 - np.radians(elevation) # angle from x-z plane up towards y axis
                    
                    
                    camera_position = [ object_center[0] + radius * np.sin(beta) * np.cos(alpha),
                                        object_center[1] + radius * np.cos(beta),
                                        object_center[2] + radius * np.sin(beta) * np.sin(alpha)]
                    camera_position = np.array(camera_position)
                    camera_quat = look_at_quaternion(camera_position, object_center)
                    cam_positions_and_quats.append((camera_position, camera_quat))

            ## Need to rightfully choose based on how many critical and diverse views we want
            threshold_dist = 1.5
            selected_views = []
            remaining_view_idxs = range(len(sorted_views))
            

            while len(remaining_view_idxs) > 0:
                selected_views.append(remaining_view_idxs[0])
                selected_view = selected_views[-1]
                views_to_be_removed = []
                for view in remaining_view_idxs:
                    if camera_distance(cam_positions_and_quats[selected_view], 
                                    cam_positions_and_quats[view]) < threshold_dist:
                        views_to_be_removed.append(view)
                views_to_be_removed.append(selected_view)
                to_remove = set(views_to_be_removed)
                remaining_view_idxs = [v for v in remaining_view_idxs if v not in to_remove]
                #remaining_view_idxs = [v for v in remaining_view_idxs if v not in views_to_be_removed]
                
            sorted_views = np.array(sorted_views)
            selected_views = np.array(selected_views)

            with open(f'{folder_path}/critical_and_diverse_views.txt', 'w') as f:
                for selected_view in selected_views:
                    img_name = sorted_views[selected_view].split('View: ')[1].split(',')[0] + '_no_colorbar.png'
                    mask_name = f'{folder_path}/renders_pyvista/mask_with_centroids' + sorted_views[selected_view].split('View: view')[1].split(',')[0] + '.png'
                    path = f'{folder_path}/renders_pyvista/{img_name}'
                    f.write(f'{sorted_views[selected_view]}| Path: {path}| Mask view: {mask_name}\n')

                

        
                    

