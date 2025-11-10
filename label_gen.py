# Compute the best view labels for training the model------------------------------
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
from compute_caption import compute_highstress_voxels, cluster_highstress_voxels
from generate_views import read_pos_file
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import trimesh
import math
from tqdm import tqdm
from datetime import datetime


def visualize_centroids_on_mask(mask_clusters, predicted_centroids, actual_centroids, out_path="mask_with_centroids.png"):
    """
    Draws large, clearly visible colored markers and labels at centroid pixel coordinates.
    mask_clusters: 2D binary or grayscale numpy array
    centroids: list of (col, row) tuples  (OpenCV expects (x=col, y=row))
    """
    # Ensure mask is 3-channel
    if len(mask_clusters.shape) == 2:
        mask_rgb = cv2.cvtColor(mask_clusters, cv2.COLOR_GRAY2BGR)
    else:
        mask_rgb = mask_clusters.copy()

    # normalize to 0–255 uint8
    mask_rgb = (mask_rgb / mask_rgb.max() * 255).astype(np.uint8)

    np.random.seed(42)
    #colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(centroids))]
    ## Green color for predicted centroids
    colors_pred = [(0,255,0) for _ in range(len(predicted_centroids))]
    ## Red color for actual centroids
    colors_actual = [(0,0,255) for _ in range(len(actual_centroids))]

    for centroids, colors in [(predicted_centroids, colors_pred), (actual_centroids, colors_actual)]:
        for i, (col, row) in enumerate(centroids):
            color = colors[i]

            if i==0 and colors == colors_pred:
                # Larger dot (radius=10)
                cv2.circle(mask_rgb, (int(col), int(row)), radius=10, color=color, thickness=-1)
            else:
                # Star dot for better visibility
                cv2.drawMarker(mask_rgb, (int(col), int(row)), color=color, markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            #cv2.circle(mask_rgb, (int(col), int(row)), radius=10, color=color, thickness=-1)
            # Bold, larger label text
            cv2.putText(mask_rgb,
                        f"{i+1}",
                        (int(col) + 14, int(row) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,        # larger font scale
                        color,
                        2,          # thicker line for visibility
                        cv2.LINE_AA)

    cv2.imwrite(out_path, mask_rgb)
    print(f"✅ Saved mask with large centroid markers to: {out_path}")
    return mask_rgb


# ---------------------------------------------------------------------
def detect_bright_clusters_and_centroids(image_path, thresh, eps, min_samples):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    # normalize to 0..1
    gray_norm = cv2.normalize(gray_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
    mask = (gray_norm > thresh).astype(np.uint8)

    coords = np.column_stack(np.where(mask > 0))  # (row, col)

    centroids = []
    mask_clusters = np.zeros_like(gray, dtype=np.uint8)

    ## Search best cluster parameters for DBSCAN
    best_score = -1
    best_eps, best_min_s = None, None

    # for eps in eps_range:
    #     for min_s in min_samples_range:
    #         labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(coords)
    #         if len(set(labels)) > 1:
    #             score = silhouette_score(coords, labels)
    #             if score > best_score:
    #                 best_score = score
    #                 best_eps, best_min_s = eps, min_s

    if coords.shape[0] > 0:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        for cid in np.unique(labels):
            if cid == -1:
                continue
            pts = coords[labels == cid]  # rows and cols
            # Fill mask for this cluster
            for (r, c) in pts:
                mask_clusters[r, c] = 255
            centroid = np.mean(pts, axis=0)  # (row, col) float
            # convert to pixel coords (u, v) convention: u = col, v = row
            v_row, u_col = centroid
            centroids.append((float(u_col), float(v_row))) # centriods in col, row format
    return img, gray, mask_clusters, centroids


def save_mask(mask, out_path):
    cv2.imwrite(out_path, mask)
    print(f"Saved mask to {out_path}")

def load_mesh(obj_path):
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Mesh not found: {obj_path}")
    mesh = trimesh.load(obj_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # If file contains multiple geometries, try to combine into single Trimesh
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh

def pixel_to_world_coords(cam_pos, F_pos, u, v, up_cam_vec, img_H, img_W, fov_in_degrees=75, mesh=None):
    ## Compute camera coordinate system
    z_cam = F_pos - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(up_cam_vec, -z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(-z_cam, x_cam)

    plane_depth = np.linalg.norm(F_pos - cam_pos)

    ## Compute cam intrinsics
    fov_y = fov_in_degrees  # degrees
    fov_y_rad = math.radians(fov_y)
    fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
    fx = fy * img_W/img_H
    cx = img_W / 2.0
    cy = img_H / 2.0

    # Compute world distance on the image plane perpendicular to the view vector
    # located at the object center i.e focal point
    x_plane = (u - cx) / fx * plane_depth
    y_plane =-(v - cy) / fy * plane_depth

    #P_world lies on the same plane as the camera center
    ray_origin = cam_pos + x_plane * x_cam + y_plane * y_cam #+ plane_depth * z_cam
    

    if mesh is None:
        P_world = ray_origin + plane_depth * z_cam
        return P_world

    ## Start a ray from this point in the view direction  
    ## Intersect this ray with the mesh to get the accurate 3D point
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=np.array([ray_origin]),
        ray_directions=np.array([z_cam])
        )

    if len(locations) ==0:
        P_world = ray_origin + plane_depth * z_cam
        return P_world
    
    P_world = locations[np.argmin(np.linalg.norm(locations - ray_origin, axis=1))]  # closest intersection point to ray start
    

    return P_world

def World_to_pixel_coords(P_world, cam_pos, F_pos, up_cam_vec, img_H, img_W, fov_in_degrees=75):
    ## Compute camera coordinate system
    z_cam = F_pos - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(up_cam_vec, -z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(-z_cam, x_cam)

    ## Compute cam intrinsics
    fov_y = fov_in_degrees  # degrees
    fov_y_rad = math.radians(fov_y)
    fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
    fx = fy * img_W/img_H
    cx = img_W / 2.0
    cy = img_H / 2.0

    ## Project P_world onto image plane
    vec_P_cam = P_world - cam_pos
    x_cam_coord = np.dot(vec_P_cam, x_cam)
    y_cam_coord = np.dot(vec_P_cam, y_cam)
    z_cam_coord = np.dot(vec_P_cam, z_cam)

    u = (fx * x_cam_coord) / z_cam_coord + cx
    v = cy - (fy * y_cam_coord) / z_cam_coord

    return (u,v)


import numpy as np

# Try to use the Hungarian solver; fallback to greedy if unavailable
try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN_AVAILABLE = True
except Exception:
    _HUNGARIAN_AVAILABLE = False

def pairwise_distances(pred, targ):
    """pred: (Np,3), targ: (Nt,3) -> (Np, Nt) distances"""
    if len(pred)==0 or len(targ)==0:
        return np.zeros((len(pred), len(targ)))
    pred = np.asarray(pred)
    targ = np.asarray(targ)
    d2 = np.sum(pred**2, axis=1)[:,None] + np.sum(targ**2, axis=1)[None,:] - 2*pred.dot(targ.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

def hungarian_match(dist_matrix):
    """Return arrays of matched indices (pred_idx, targ_idx) using Hungarian (min-cost).
       If scipy not available, use greedy matching.
    """
    if dist_matrix.size == 0:
        return np.array([],dtype=int), np.array([],dtype=int)
    if _HUNGARIAN_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)
    # greedy fallback (not optimal but usable)
    d = dist_matrix.copy()
    n,p = d.shape
    matched_pred, matched_targ = [], []
    used_r = set()
    used_c = set()
    # flatten order by smallest distance first
    inds = np.dstack(np.unravel_index(np.argsort(d.ravel()), d.shape))[0]
    for r,c in inds:
        if r in used_r or c in used_c:
            continue
        matched_pred.append(r); matched_targ.append(c)
        used_r.add(r); used_c.add(c)
        if len(used_r) == n or len(used_c) == p:
            break
    return np.array(matched_pred, dtype=int), np.array(matched_targ, dtype=int)

def evaluate_centroids(predicted, target, match_threshold=None, return_details=False):
    """
    Evaluate predicted vs target 3D centroids.

    predicted: (Np,3) list/array of predicted points
    target:    (Nt,3) list/array of ground-truth points
    match_threshold: float or None. If provided, matches with distance <= threshold are TP.
                     Otherwise all Hungarian matches are considered matched and distances reported.
    Returns dict with metrics:
      - TP, FP, FN
      - precision, recall, f1
      - mean_matched_dist, median_matched_dist, rmse_matched_dist
      - chamfer_dist (symmetric average nearest-neighbor)
      - hausdorff_dist (symmetric max of nearest-neighbor)
    """
    pred = np.asarray(predicted).reshape(-1,3)
    targ = np.asarray(target).reshape(-1,3)

    Np = len(pred); Nt = len(targ)

    result = {"Np":Np, "Nt":Nt}

    if Np == 0 and Nt == 0:
        # perfect empty case
        result.update({"TP":0,"FP":0,"FN":0,"precision":1.0,"recall":1.0,"f1":1.0,
                       "mean_matched_dist":0.0,"median_matched_dist":0.0,"rmse_matched_dist":0.0,
                       "chamfer":0.0,"hausdorff":0.0})
        return (result if not return_details else (result, {}))

    if Np == 0:
        result.update({"TP":0,"FP":0,"FN":Nt,"precision":0.0,"recall":0.0,"f1":0.0})
        # compute chamfer/hausdorff: distances from targ->pred are inf or large
        # We'll return inf for hausdorff and chamfer as mean nearest neighbor
    if Nt == 0:
        result.update({"TP":0,"FP":Np,"FN":0,"precision":0.0,"recall":0.0,"f1":0.0})

    D = pairwise_distances(pred, targ)  # shape (Np, Nt)

    # Hungarian matching (gives best 1-1 pairing, possibly matching all when counts differ)
    row_ind, col_ind = hungarian_match(D)

    # distances for matched pairs
    if row_ind.size>0:
        matched_dists = D[row_ind, col_ind]
    else:
        matched_dists = np.array([])

    # If threshold given, decide TP/FP/FN
    if match_threshold is not None:
        # TP = matched pairs with distance <= threshold
        good = matched_dists <= match_threshold
        TP = int(np.count_nonzero(good))
        FP = int(Np - TP)
        FN = int(Nt - TP)
        matched_pairs = list(zip(row_ind[good], col_ind[good]))
    else:
        # Count all matched pairs as matches (for reporting)
        TP = int(len(row_ind))
        FP = int(max(0, Np - TP))
        FN = int(max(0, Nt - TP))
        matched_pairs = list(zip(row_ind, col_ind))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision+recall>0 else 0.0

    # matched distance stats (only for true positive matches if threshold used, otherwise for all matched pairs)
    if match_threshold is not None:
        d_stats_arr = matched_dists[matched_dists <= match_threshold] if matched_dists.size>0 else np.array([])
    else:
        d_stats_arr = matched_dists

    if d_stats_arr.size>0:
        mean_d = float(np.mean(d_stats_arr))
        median_d = float(np.median(d_stats_arr))
        rmse_d = float(np.sqrt(np.mean(d_stats_arr**2)))
    else:
        mean_d = median_d = rmse_d = float("nan")

    # Chamfer: average of nearest neighbor distances both directions
    if Np > 0 and Nt > 0:
        # pred -> targ
        nn_pred_to_targ = np.min(D, axis=1) if D.size>0 else np.array([])
        # targ -> pred
        nn_targ_to_pred = np.min(D, axis=0) if D.size>0 else np.array([])
        chamfer = 0.5 * (float(np.mean(nn_pred_to_targ)) + float(np.mean(nn_targ_to_pred)))
        hausdorff = float(max(np.max(nn_pred_to_targ), np.max(nn_targ_to_pred)))
    else:
        chamfer = float("inf")
        hausdorff = float("inf")

    result.update({
        "TP":TP, "FP":FP, "FN":FN,
        "precision":precision, "recall":recall, "f1":f1,
        "mean_matched_dist":mean_d, "median_matched_dist":median_d, "rmse_matched_dist":rmse_d,
        "chamfer":chamfer, "hausdorff":hausdorff,
        "matched_pairs": matched_pairs
    })
    if return_details:
        details = {"D":D, "row_ind":row_ind, "col_ind":col_ind, "matched_dists":matched_dists}
        return result, details
    return result


# ---------------------- main pipeline --------------------------------
def main():

    main_geom_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries'
    
    pos_files_processed = set()
    fname_pos = 'pos_files_accounted_for_labelgen.txt'
    if os.path.exists(fname_pos):
        with open(fname_pos, 'r') as f:
            for line in f:
                pos_files_processed.add(line.strip())
    # ---------- User inputs / replace with real values ----------
    for cad_folder in tqdm(os.listdir(main_geom_dir)):
        folder_path = os.path.join(main_geom_dir, cad_folder)
        if not os.path.isdir(folder_path):
            continue
        if not os.path.exists(os.path.join(folder_path, 'renders_pyvista')): continue
        
        POS_PATH   = f"{folder_path}/{cad_folder}_grad_0.pos"         # path to your POS
        OBJ_PATH   = f"{folder_path}/renders_pyvista/{cad_folder}.obj"
        if not os.path.exists(POS_PATH):
            print(f"pos file not exist: {POS_PATH}")
            continue
        if POS_PATH in pos_files_processed:
            print(f"Skipping already processed POS file: {POS_PATH}")
            continue
        try:
            active_voxels = compute_highstress_voxels(POS_PATH, voxel_resolution=9, threshold=0.3)
        except: 
            print(f'Invalid file {POS_PATH}')
            continue    
        target_clusters = cluster_highstress_voxels(active_voxels=active_voxels, voxel_resolution=9)
        target_centroids = [cluster['centroid'] for cluster in target_clusters]

        obj_mesh = trimesh.load(OBJ_PATH)
        
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(fname_pos, 'a') as f:
            f.write(f'{POS_PATH}\n')

        coords, vals = read_pos_file(POS_PATH)
        object_center = coords.mean(axis=0)
        zoom = 4.5
        radius = np.linalg.norm(coords - object_center, axis=1).max() * zoom
        save_img_res = (1000, 1000)
        eval_results_CAD = []
        for img_path in os.listdir(os.path.join(folder_path, 'renders_pyvista')):
            if not img_path.endswith('no_colorbar.png'):
                continue
            IMAGE_PATH = os.path.join(folder_path, 'renders_pyvista', img_path)
        
            folder_path, cad_folder = '/'.join(IMAGE_PATH.split('/')[:-2]), IMAGE_PATH.split('/')[-3]
            
            ## Azimuth and elevation from the image filename
            azimuth =  int(IMAGE_PATH.split('/')[-1].split('.')[0].split('_a')[1].split('_')[0])
            elevation = int(IMAGE_PATH.split('/')[-1].split('.')[0].split('_e')[1].split('_')[0])

            # Convert spherical coordinates to Cartesian camera position
            ## z-axis in pyvista points is outward from screen (Right hand rule i.e not inward and hence the revolving angle alpha=np.pi/2-azimuth)
            ## elevation is the angle from the +y axis down toward the x-z plane
            ## azimuth is the angle taken from the positive x-axis toward the positive z-axis
            alpha = np.pi/2 - np.radians(azimuth) # angle from z axis towards +x axis in the azimuthal plane y=0
            beta =  np.pi/2 - np.radians(elevation) # angle from x-z plane up towards y axis
                    
            camera_position = [ object_center[0] + radius * np.sin(beta) * np.cos(alpha),
                                object_center[1] + radius * np.cos(beta),
                                object_center[2] + radius * np.sin(beta) * np.sin(alpha)]

            img = cv2.imread(IMAGE_PATH)
            img = cv2.resize(img, save_img_res)
            cv2.imwrite(IMAGE_PATH, img)
            img_height, img_width = img.shape[0:2]
            
            # DBSCAN / bright threshold parameters (tune to your image)
            BRIGHTNESS_THRESHOLD = 0.6   # pixel intensity normalized [0,1]
            
            DBSCAN_EPS= 20 #np.arange(1,5) * 10              # in pixels
            DBSCAN_MIN_SAMPLES = 70 # np.arange(5,15) * 10     # min samples per cluster

            # Detect centroids (col, row)
            img, gray, mask_clusters, pixel_centroids = detect_bright_clusters_and_centroids(IMAGE_PATH, thresh=BRIGHTNESS_THRESHOLD, eps= DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            # img, gray, mask_clusters, pixel_centroids = detect_bright_clusters_kmeans(IMAGE_PATH, thresh=BRIGHTNESS_THRESHOLD, k_range=range(2,10))
            print(f"Detected {len(pixel_centroids)} centroids from image.")

            actual_centroids = []
            for target_centroid in target_centroids:
                P_world = np.array(target_centroid)
                u,v = World_to_pixel_coords(P_world, np.array(camera_position), object_center,
                                            up_cam_vec=np.array([0,1,0]),
                                            img_H=img_height, img_W=img_width, fov_in_degrees=30)
                actual_centroids.append((u,v))
            mask_with_centroids = visualize_centroids_on_mask(mask_clusters, predicted_centroids=pixel_centroids, actual_centroids=actual_centroids,
                                                        out_path=os.path.join(folder_path, 'renders_pyvista', f'mask_with_centroids_e{elevation}_a{azimuth}.png'))

            predicted_centroids = []
            for pixel_centroid in pixel_centroids:
                u, v = pixel_centroid
                P_world = pixel_to_world_coords(np.array(camera_position), object_center,
                                                u, v, up_cam_vec=np.array([0,1,0]),
                                                img_H=img_height, img_W=img_width, fov_in_degrees=30, mesh=obj_mesh)
                predicted_centroids.append(P_world)
            
            ## Evaluating how accurate the view is from predicted_centroids and target_centroids
            match_distance = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) * 0.1  # 10% of bounding box diagonal
            eval_results, eval_details = evaluate_centroids(predicted_centroids, target_centroids,
                                                            match_threshold=match_distance, return_details=True)

            eval_results_CAD.append({f'view_e{elevation}_a{azimuth}': eval_results})
            
        ## Sort metric first on F1 score (descending) and then chamfer distance (ascending)
        sorted_views = sorted(eval_results_CAD, key=lambda x: (-list(x.values())[0]['f1'], list(x.values())[0]['chamfer']))
        with open(f'{folder_path}/{cad_folder}_top_views.txt', 'w') as f:
            for view_dict in sorted_views:  # top 20 views
                view_name = list(view_dict.keys())[0]
                metrics = list(view_dict.values())[0]
                f.write(f"View: {view_name}, F1: {metrics['f1']:.4f}, Chamfer: {metrics['chamfer']:.4f}\n")
                print(f"View: {view_name}, F1: {metrics['f1']:.4f}, Chamfer: {metrics['chamfer']:.4f}")    
        # print(f"Top views for CAD {cad_folder}:")
        # for view_dict in sorted_views[:20]:  # top 5 views
        #     view_name = list(view_dict.keys())[0]
        #     metrics = list(view_dict.values())[0]
        #     print(f"View: {view_name}, F1: {metrics['f1']:.4f}, Chamfer: {metrics['chamfer']:.4f}")
        # print('Done with CAD folder')
    

if __name__ == "__main__":
    main()


    

