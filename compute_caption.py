import sys
import os
from tqdm import tqdm
freecad_base = "/data/1bali/miniforge3/envs/vtk_offscreen"
# Append FreeCAD's Python library paths
sys.path.append(os.path.join(freecad_base, "lib"))
sys.path.append(os.path.join(freecad_base, "Mod"))   

from generate_views import read_pos_file

import numpy as np
from scipy.ndimage import label, generate_binary_structure
from generate_views import read_pos_file

def compute_highstress_voxels(pos_file_path, voxel_resolution=8, threshold=0.3):
    """
    Compute high-stress voxel bins and return their centers (in absolute geometry coordinates)
    along with the points and values within each voxel.
    """
    coords, vals = read_pos_file(pos_file_path)

    # Get bounding box in absolute coordinates
    bbox_min, bbox_max = coords.min(0), coords.max(0)
    bbox_size = bbox_max - bbox_min

    # Map points into voxel grid indices (using absolute positions)
    voxel_indices = np.clip(((coords - bbox_min) / (bbox_size + 1e-9) * voxel_resolution).astype(int),
                            0, voxel_resolution - 1)

    # Data structure to store voxel contents
    voxel_data = {}

    for (x, y, z), val, coord in zip(voxel_indices, vals, coords):
        if val > threshold:
            key = (x, y, z)
            if key not in voxel_data:
                voxel_data[key] = {"points": [], "values": []}
            voxel_data[key]["points"].append(coord)
            voxel_data[key]["values"].append(val)

    # Compute voxel centers in absolute geometry coordinates
    active_voxels = []
    for (x, y, z), data in voxel_data.items():
        # Compute voxel center (absolute)
        voxel_center_norm = (np.array([x, y, z]) + 0.5) / voxel_resolution
        voxel_center_real = bbox_min + voxel_center_norm * bbox_size
        active_voxels.append({
            "voxel_index": (x, y, z),
            "center_coord": voxel_center_real,
            "points": np.array(data["points"]),
            "values": np.array(data["values"]),
        })

    # Sort voxels by number of high-stress points (descending)
    active_voxels = sorted(active_voxels, key=lambda v: len(v["values"]), reverse=True)

    print(f"Active voxels (> {threshold}): {len(active_voxels)}")
    return active_voxels

import numpy as np
from scipy.ndimage import label, generate_binary_structure

def classify_strength(value):
    """Classify region intensity."""
    if value > 0.9:
        return "strong"
    elif value > 0.6:
        return "moderate"
    else:
        return "mild"

def get_region_description(norm_coord):
    """Generate human-readable location description given normalized coordinate (0–1)."""
    x, y, z = norm_coord
    desc = []

    desc.append("left" if x < 0.33 else "right" if x > 0.66 else "center")
    desc.append("front" if y < 0.33 else "behind" if y > 0.66 else "mid")
    desc.append("bottom" if z < 0.33 else "top" if z > 0.66 else "mid")

    return "-".join(desc)

def cluster_highstress_voxels(active_voxels, voxel_resolution=8):
    """
    Cluster spatially adjacent high-stress voxels and merge their contents.
    Returns a list of clusters with centroids and all point data.
    """
    # Create a 3D binary mask for active voxels
    voxel_mask = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
    for v in active_voxels:
        x, y, z = v["voxel_index"]
        voxel_mask[x, y, z] = True

    # 26-connectivity for 3D (face, edge, and corner adjacency)
    struct = generate_binary_structure(3, 3)
    labeled, num_clusters = label(voxel_mask, structure=struct)

    print(f"Detected {num_clusters} voxel clusters.")

    if num_clusters == 0:
        return []

    clusters = []
    # Prepare mapping from voxel index to its data
    voxel_dict = {tuple(v["voxel_index"]): v for v in active_voxels}

    for cluster_id in range(1, num_clusters + 1):
        cluster_voxels = np.argwhere(labeled == cluster_id)
        cluster_points, cluster_values = [], []

        # Merge data from voxels in this cluster
        for x, y, z in cluster_voxels:
            key = (x, y, z)
            if key in voxel_dict:
                cluster_points.append(voxel_dict[key]["points"])
                cluster_values.append(voxel_dict[key]["values"])

        if not cluster_points:
            continue

        cluster_points = np.concatenate(cluster_points, axis=0)
        cluster_values = np.concatenate(cluster_values, axis=0)

        # Compute centroid in actual geometry coordinates
        centroid = cluster_points.mean(axis=0)

        clusters.append({
            "cluster_id": cluster_id,
            "voxels": [tuple(v) for v in cluster_voxels],
            "centroid": centroid,
            "points": cluster_points,
            "values": cluster_values,
        })

    print(f"Formed {len(clusters)} merged clusters.")
    
    clusters = sorted(clusters, key=lambda v:len(v["values"]), reverse=True) # Sort by number of points in each cluster
    return clusters

import numpy as np

def describe_highstress_clusters(clusters, pos_file_path, n_clusters=3):
    """
    Generate a caption describing the locations of the top N high-stress clusters
    relative to the object centroid.
    """
    coords, vals = read_pos_file(pos_file_path)
    object_centroid = np.mean(coords, axis=0)

    if len(clusters) == 0:
        return "No significant high-stress regions detected."

    # Sort clusters by number of values (already done usually)
    clusters = sorted(clusters, key=lambda v: len(v["values"]), reverse=True)[:n_clusters]

    def region_descriptor(rel):
        """Map relative centroid position to a textual region description."""
        x, y, z = rel
        desc = []

        # X axis (left-right)
        if x > 0.15:
            desc.append("right")
        elif x < -0.15:
            desc.append("left")

        # z axis (front-behind)
        if z > 0.15:
            desc.append("front")
        elif z < -0.15:
            desc.append("behind")

        # y axis (top-bottom)
        if y > 0.15:
            desc.append("top")
        elif y < -0.15:
            desc.append("bottom")

        if not desc:
            return "center"
        return "-".join(desc)

    captions = []
    for cluster in clusters:
        centroid = cluster["centroid"]
        rel = centroid - object_centroid

        # Normalize relative position to [-1,1]
        rel_norm = rel / (np.max(np.abs(rel)) + 1e-9)
        region = region_descriptor(rel_norm)

        # # Optionally classify intensity
        # val = np.mean(cluster["values"])
        # if val > 0.8:
        #     strength = "strong"
        # elif val > 0.5:
        #     strength = "moderate"
        # else:
        #     strength = "mild"

        captions.append(region)

    # Merge into sentence
    caption = 'high stress gradients at '
    caption+= ", ".join(captions)
    caption = caption[0].upper() + caption[1:] + "."
    return caption

if __name__ == "__main__":

    #/data/1bali/Other_LLM_projects/multi_view_3DQA/2D_FE_Mesh_2.jpg
    path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries"
    CAD_mesh_path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-meshes"
    processed_files_, timeout_files_ = [], []
    for CAD_Folder in tqdm(os.listdir(path_dir)):
        
        ## Render views from the simulated pos file
        pos_file_path = f'{path_dir}/{CAD_Folder}/{CAD_Folder}_grad_0.pos'
        if not os.path.exists(pos_file_path):
            print(f"pos file does not exist: {pos_file_path}")
            continue
        #output_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista'

        caption_file = f'{path_dir}/{CAD_Folder}/rule_based_caption.txt'
        try:
            active_voxels = compute_highstress_voxels(pos_file_path)

            clusters = cluster_highstress_voxels(active_voxels=active_voxels)

            caption = describe_highstress_clusters(clusters=clusters, pos_file_path=pos_file_path)

            with open(caption_file, 'w') as fout:
                fout.write(caption)
            print(f'Written {caption_file}')
        except:
            print(f'Could not make it for {caption_file}')