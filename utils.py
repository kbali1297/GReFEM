api_key = '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'
from collections import defaultdict
import os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
import numpy as np
import math
import trimesh
import cv2
import pyvista as pv
import gmsh
import meshio
from tqdm import tqdm
from general_prompts import *
import requests
import base64
from generate_renders import render_mesh_views, render_mesh_views_with_load

def get_perpendicular_views(elev_deg, azim_deg):
    """
    Given an input (elev, azim), returns 4 perpendicular views 
    lying in the plane defined by the input direction as the normal.
    """
    # 1. Convert input to Cartesian (using your code's convention)
    # x = cos(e) sin(a)
    # y = sin(e)
    # z = -cos(e) cos(a)
    e_rad = np.radians(elev_deg)
    a_rad = np.radians(azim_deg)
    
    v = np.array([
        np.cos(e_rad) * np.sin(a_rad),
        np.sin(e_rad),
        -np.cos(e_rad) * np.cos(a_rad)
    ])
    v = v / np.linalg.norm(v)

    # 2. Find a starting perpendicular vector (Basis Vector 1)
    # Use global up (0, 1, 0). If v is already up, use (1, 0, 0)
    up = np.array([0, 1, 0])
    if abs(np.dot(v, up)) > 0.99:
        up = np.array([1, 0, 0])
        
    u1 = np.cross(up, v)
    u1 /= np.linalg.norm(u1)
    
    # 3. Find the other perpendicular basis vector in the plane
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)

    # 4. Generate 4 vectors in the plane at 0, 90, 180, 270 degrees
    # P = cos(theta)*u1 + sin(theta)*u2
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    perpendicular_pairs = []

    for theta in angles:
        p = np.cos(theta) * u1 + np.sin(theta) * u2
        
        # 5. Convert Cartesian back to (elev, azim)
        # elev = arcsin(y)
        # azim = atan2(x, -z)
        new_elev = np.degrees(np.arcsin(np.clip(p[1], -1.0, 1.0)))
        new_azim = np.degrees(np.arctan2(p[0], -p[2]))
        
        # Normalize azimuth to [0, 360)
        new_azim = (new_azim + 360) % 360
        
        perpendicular_pairs.append((int(round(new_elev)), int(round(new_azim))))

    return perpendicular_pairs

def angle_btw_views(view1, view2):
    el1, az1 = view1
    el2, az2 = view2
    # Convert to radians
    el1_rad, az1_rad = math.radians(el1), math.radians(az1)
    el2_rad, az2_rad = math.radians(el2), math.radians(az2)
    
    # Convert spherical to Cartesian coordinates (assuming radius=1 for direction vectors)
    x1 = math.cos(el1_rad) * math.cos(az1_rad)
    y1 = math.cos(el1_rad) * math.sin(az1_rad)
    z1 = math.sin(el1_rad)
    
    x2 = math.cos(el2_rad) * math.cos(az2_rad)
    y2 = math.cos(el2_rad) * math.sin(az2_rad)
    z2 = math.sin(el2_rad)
    
    # Compute dot product and magnitudes
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    mag_v1 = math.sqrt(x1**2 + y1**2 + z1**2)
    mag_v2 = math.sqrt(x2**2 + y2**2 + z2**2)
    
    # Compute angle in degrees
    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # Avoid division by zero, treat as same view
    
    cos_angle = dot_product / (mag_v1 * mag_v2)
    cos_angle = max(min(cos_angle, 1), -1)  # Clamp to valid range
    angle_deg = math.degrees(math.acos(cos_angle))
    
    return angle_deg

from scipy.spatial import KDTree
import numpy as np

def filter_points_by_density_fast(points, point_density):
    """
    High-speed spatial thinning using a KDTree.
    """
    if len(points) == 0:
        return np.array([])

    points = np.atleast_2d(points)
    kept_indices = []
    # We'll use a set to keep track of indices to ignore
    ignored_mask = np.zeros(len(points), dtype=bool)
    
    tree = KDTree(points)

    for i in range(len(points)):
        if ignored_mask[i]:
            continue
            
        # Keep this point
        kept_indices.append(i)
        
        # Find all points within 'point_density' of this point and mark them to be ignored
        # This prevents them from being added to kept_indices later
        neighbors = tree.query_ball_point(points[i], point_density)
        ignored_mask[neighbors] = True
        
    return points[kept_indices]

def remove_points_by_proximity(points_3d, points_to_remove, point_density):
    if len(points_to_remove) == 0:
        return np.array(points_3d)

    # 1. Build a search tree of the points we want to avoid
    remove_tree = KDTree(points_to_remove)

    # 2. For every point in points_3d, find how many points from 
    # the "remove set" are within the 'point_density' distance.
    # query_ball_point returns a list of indices for each point.
    indices = remove_tree.query_ball_point(points_3d, r=point_density)

    # 3. If the list is empty, it means NO points were nearby. We keep those.
    keep_mask = [len(idx) == 0 for idx in indices]
    
    return np.array(points_3d)[keep_mask]

def pixel_to_mesh(cam_pos, F_pos, u, v, up_cam_vec, img_H, img_W,
                  fov_in_degrees=75, parallel_scale=None, mesh=None,
                  orthographic=False, return_all_hits=False, return_no_location=False):
    """
    Vectorized pixel_to_mesh. 
    u and v can be single scalar floats or lists/numpy arrays of pixel coordinates.
    """
    is_scalar = np.isscalar(u)
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    N = len(u)

    # -------------------------------
    # Camera coordinate system
    # -------------------------------
    z_cam = F_pos - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(up_cam_vec, -z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(-z_cam, x_cam)

    cx = img_W / 2.0
    cy = img_H / 2.0
    plane_depth = np.linalg.norm(F_pos - cam_pos)

    # -------------------------------
    # PERSPECTIVE PROJECTION
    # -------------------------------
    if not orthographic:
        fov_y_rad = math.radians(fov_in_degrees)
        fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
        fx = fy * img_W / img_H

        x_plane = (u - cx) / fx * plane_depth
        y_plane = -(v - cy) / fy * plane_depth

        # Vectorized generation of N ray origins and directions
        ray_origins = cam_pos + np.outer(x_plane, x_cam) + np.outer(y_plane, y_cam)
        ray_directions = np.tile(z_cam, (N, 1))

    # -------------------------------
    # ORTHOGRAPHIC PROJECTION
    # -------------------------------
    else:
        if parallel_scale is None:
            raise ValueError("parallel_scale must be provided for orthographic projection")

        world_height = 2.0 * parallel_scale
        world_width = world_height * img_W / img_H

        sx = world_width / img_W
        sy = world_height / img_H

        dx = (u - cx) * sx
        dy = -(v - cy) * sy

        P_plane = F_pos + np.outer(dx, x_cam) + np.outer(dy, y_cam)
        ray_origins = P_plane + 1000000.0 * z_cam
        ray_directions = np.tile(-z_cam, (N, 1))

    # -------------------------------
    # Ray–mesh intersection
    # -------------------------------
    if mesh is None:
        fallback = ray_origins + plane_depth * z_cam
        results = [f[None, :] if return_all_hits else f for f in fallback]
        return results[0] if is_scalar else results

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )

    # Group hits by ray index
    hits_per_ray = [[] for _ in range(N)]
    for loc, ray_idx in zip(locations, index_ray):
        hits_per_ray[ray_idx].append(loc)

    results = []
    for i in range(N):
        if len(hits_per_ray[i]) == 0:
            if return_no_location:
                results.append(None)
            else:
                results.append(np.array([ray_origins[i] + plane_depth * z_cam]))
        else:
            hits = np.array(hits_per_ray[i])
            # sort hits by distance to camera
            dists = np.linalg.norm(hits - cam_pos, axis=1)
            hits = hits[np.argsort(dists)]
            
            if return_all_hits:
                # To match previous shape logic: return array of hits
                results.append(hits if len(hits.shape) > 1 else hits[None, :])
            else:
                results.append(hits[0])

    return results[0] if is_scalar else results


def space_to_pixel(points_3d, cam_pos, F_pos, up_cam_vec, img_H, img_W,
                  fov_in_degrees=75, parallel_scale=None, orthographic=False):
    """
    Projects 3D points into 2D pixel coordinates (u, v).
    Inverse of the pixel_to_mesh logic.
    
    Args:
        points_3d: (N, 3) array or list of 3D points.
        cam_pos, F_pos, up_cam_vec: Camera vectors.
        img_H, img_W: Image resolution.
        fov_in_degrees: For perspective projection.
        parallel_scale: For orthographic projection.
        orthographic: Boolean toggle.
        
    Returns:
        u, v: Numpy arrays of pixel coordinates.
    """
    points_3d = np.atleast_2d(points_3d)
    N = points_3d.shape[0]

    # 1. Recreate the same Camera Coordinate System (Basis Vectors)
    z_cam = F_pos - cam_pos
    dist_to_plane = np.linalg.norm(z_cam)
    z_cam = z_cam / dist_to_plane
    
    x_cam = np.cross(up_cam_vec, -z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    
    y_cam = np.cross(-z_cam, x_cam)

    # 2. Translate points to camera local space
    # Move origin to camera position
    pts_rel = points_3d - cam_pos

    # 3. Project points onto camera basis vectors (Dot products)
    # This transforms world coords into local (X, Y, Z) camera space
    # X: Right, Y: Up, Z: Forward
    X = np.sum(pts_rel * x_cam, axis=1)
    Y = np.sum(pts_rel * y_cam, axis=1)
    Z = np.sum(pts_rel * z_cam, axis=1)

    cx = img_W / 2.0
    cy = img_H / 2.0

    if not orthographic:
        # --- PERSPECTIVE PROJECTION ---
        fov_y_rad = math.radians(fov_in_degrees)
        fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
        fx = fy * img_W / img_H

        # Project using Z depth
        # Note: Avoid division by zero for points behind/at camera
        Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        
        u = (X / Z_safe) * fx + cx
        v = cy - (Y / Z_safe) * fy

    else:
        # --- ORTHOGRAPHIC PROJECTION ---
        if parallel_scale is None:
            raise ValueError("parallel_scale must be provided for orthographic projection")

        world_height = 2.0 * parallel_scale
        world_width = world_height * img_W / img_H

        sx = world_width / img_W
        sy = world_height / img_H

        # In orthographic, u,v are linear mappings of local X,Y
        u = (X / sx) + cx
        v = cy - (Y / sy)

    pixel_coords = np.stack([u, v], axis=1)
    return pixel_coords

def check_points_visibility(mesh, cam_pos, points, tol=1e-4):
    if len(points) == 0:
        return np.array([]), None
    ray_vecs = points - cam_pos
    ray_dist = np.linalg.norm(ray_vecs, axis=1)

    ray_dirs = ray_vecs / ray_dist[:, None]

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=np.tile(cam_pos[None, :], (len(points), 1)),
        ray_directions=ray_dirs
    )

    # distance of each intersection from camera
    hit_dist = np.linalg.norm(locations - cam_pos, axis=1)

    # initialize closest hit distance per ray
    closest_hit = np.full(len(points), np.inf)

    # keep minimum distance per ray
    np.minimum.at(closest_hit, index_ray, hit_dist)

    # visible if closest hit ≈ target distance
    visible_idxs = np.abs(closest_hit - ray_dist) < tol
    avg_distance_to_visible = np.mean(ray_dist[visible_idxs]) if np.sum(visible_idxs) > 0 else None
    return visible_idxs, avg_distance_to_visible

def is_pixel_in_pred_cells(pixel, pred_cells, gridx, gridy, img_shape):
    img_H, img_W = img_shape[:2]
    cell_width = img_W / gridx
    cell_height = img_H / gridy

    for cell_num in pred_cells:
        r = (cell_num - 1) // gridx
        c = (cell_num - 1) % gridx

        x0 = c * cell_width
        x1 = (c + 1) * cell_width
        y0 = r * cell_height
        y1 = (r + 1) * cell_height

        if x0 <= pixel[0] < x1 and y0 <= pixel[1] < y1:
            return True

    return False

def calculate_dynamic_sampling_params(point_density, img_shape, cam_pos, F_pos, 
                                      fov_deg=75, parallel_scale=None, orthographic=True):
    # 1. Compute 3D world density
    point_density_3d = point_density
    img_H = img_shape[0]
    
    # 2. Compute World Units Per Pixel (sy)
    if orthographic:
        # sy = world_height / img_H
        world_height = 2.0 * parallel_scale
        sy = world_height / img_H
    else:
        # sy = dist_to_target / focal_length_in_pixels
        dist = np.linalg.norm(cam_pos - F_pos)
        fy = 0.5 * img_H / np.tan(np.radians(fov_deg) * 0.5)
        sy = dist / fy

    # 3. Derive sampling_step (how far apart points are along the contour)
    # We want a sample roughly every 'point_density_3d' units in world space
    sampling_step = max(2, int(point_density_3d / sy))
    
    # 4. Derive pixel_offset (how far p_in/p_out are from p_on)
    # This should be small enough to stay near the edge but large enough 
    # to register a depth difference. 1/3 of the density is usually a good heuristic.
    pixel_offset = max(1, int(sampling_step))#/3
    
    return sampling_step, pixel_offset




def overlay_grid(image_path, gridx=10, gridy=10, extra_folder_specs="", font_scale=1.0, folder_path=None, font_color=(255,255,255), grid_color=(255,255,255), font_thickness=2, line_thickness=2, font = cv2.FONT_HERSHEY_SIMPLEX, arrow_cell=None, arrow_color=(255,0,0), arrow_thickness=2, arrow_len_cells=1, output_dir=None):
    
    if folder_path==None:
        folder_path = '/'.join(image_path.split('/')[:-2])
    n_rows, n_cols = gridy, gridx              # grid resolution
    
    if extra_folder_specs != "":
        grid_img_dirpath = f'{folder_path}/renders_grid_{extra_folder_specs}_{n_rows}X{n_cols}'
    else:
        grid_img_dirpath = f'{folder_path}/renders_grid_{n_rows}X{n_cols}'

    os.makedirs(grid_img_dirpath, exist_ok=True)
    img_name = image_path.split('/')[-1]
    if output_dir is None:
        output_path = f'{grid_img_dirpath}/{img_name}'
    else:
        os.makedirs(f'{output_dir}/renders_grid_{n_rows}X{n_cols}', exist_ok=True)
        output_path = f'{output_dir}/renders_grid_{n_rows}X{n_cols}/{img_name}'
    
    scale_factor = 100/(n_cols * n_rows)
    font_scale *= scale_factor                   # label size

    # ==== LOAD IMAGE ====
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # ==== DRAW GRID ====
    cell_h = h // n_rows
    cell_w = w // n_cols

    number = 1
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            # Draw cell rectangle
            cv2.rectangle(img, (x0, y0), (x1, y1), grid_color, line_thickness)

            # Center coordinates for text
            text = str(number)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x0 + (cell_w - text_size[0]) // 2
            text_y = y0 + (cell_h + text_size[1]) // 2

            # Draw text with outline for visibility
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 3, cv2.LINE_AA)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            number += 1

    # ==== DRAW DOWNWARD ARROW ====
    if arrow_cell is not None:
        r = (arrow_cell - 1)// n_rows
        c = (arrow_cell - 1)% n_cols
        
        # Tip of arrow at cell center
        tip_x = int(c * cell_w + cell_w / 2)
        tip_y = int(r * cell_h + cell_h / 2)

        # Arrow starts above and points downward
        start_x = tip_x
        start_y = int(tip_y - arrow_len_cells * cell_h)

        cv2.arrowedLine(
            img,
            (start_x, start_y),
            (tip_x, tip_y),
            arrow_color,
            arrow_thickness,
            tipLength=0.3
        )
    # ==== SAVE OUTPUT ====
    cv2.imwrite(output_path, img)
    return output_path
    #print(f"Saved numbered image as {output_path}")

def detect_corner_by_avg(cnt, x0, y0, window=6, angle_thresh_deg=20):
    import numpy as np

    # flatten contour
    pts = cnt[:, 0, :].astype(np.float64)

    # Make the contour continuous by appending the first few points at the end
    pts = np.vstack([pts, pts])

    ## if lot of distance between two consecutive points, seperate into multiple point lists
    pts_ = pts[1:] - pts[:-1]
    dists = np.linalg.norm(pts_, axis=1)
    split_indices = np.where(dists > 2)[0] + 1
    if len(split_indices) > 0:
        pts_list = np.split(pts, split_indices, axis=0)
    else:
        pts_list = [pts]

    corners = []
    for pts in pts_list:
        # step vectors
        vecs = pts[1:] - pts[:-1]

        # normalize non-zero vectors
        norms = np.linalg.norm(vecs, axis=1)
        vecs = vecs[norms > 0]
        norms = norms[norms > 0]
        vecs /= norms[:, None]

        for i in range(window, len(vecs) - window):
            v_prev = vecs[i-window:i].mean(axis=0)
            v_next = vecs[i:i+window].mean(axis=0)

            # normalize averages
            if np.linalg.norm(v_prev) < 1e-6 or np.linalg.norm(v_next) < 1e-6:
                continue

            v_prev /= np.linalg.norm(v_prev)
            v_next /= np.linalg.norm(v_next)

            # angle between dominant directions
            cosang = np.clip(np.dot(v_prev, v_next), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosang))

            if angle > angle_thresh_deg:
                cx, cy = pts[i]
                #return int(cx + x0), int(cy + y0)
                corners.append((int(cx + x0), int(cy + y0)))
                break  # only one corner per contour segment
    
    
    return corners

import cv2
import numpy as np

def edge_detection_contour(image_path, cell_num, gridx=10, gridy=10, angle_thresh=30):
    """
    image_path: path to the image file
    cell_num: 1-based cell number in the grid
    gridx, gridy: number of grid cells in x and y directions
    Returns (x, y) of a real corner inside the given grid cell.
    Returns None if no real corner is present.
    """
    min_run = 5  # minimum number of consecutive points in same direction to consider a persistent change
    # Load image and binary mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 240, 1, cv2.THRESH_BINARY_INV)

    h, w = img.shape

    # Grid indexing (1-based cell_num)
    r = (cell_num - 1) // gridx
    c = (cell_num - 1) % gridx

    x0 = c * (w // gridx)
    x1 = (c + 1) * (w // gridx)
    y0 = r * (h // gridy)
    y1 = (r + 1) * (h // gridy)

    # Cell dimensions
    cell_width, cell_height = x1 - x0, y1 - y0

    cell_patch = img[y0:y1, x0:x1]
    
    contours, _ = cv2.findContours(cell_patch, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Filter out points from contour touching the cell border
    contours = [contour[(contour[:,0,0]!=cell_width-1) & (contour[:,0,1]!=cell_height-1) & (contour[:,0,0]!=0) & (contour[:,0,1]!=0)] for contour in contours]
    
    corners =[]
    for contour in contours:
        if len(contour) < 20:
            continue
        # Check for real corner
        corners.extend(detect_corner_by_avg(contour, x0, y0, window=4, angle_thresh_deg=angle_thresh))

    inside_corners =[]

    if corners:
        inspect_r = 6 
        num_theta = 72
        
        # 5. Pre-compute the circle offsets using endpoint=False (prevents double-counting 360 deg and 0 deg)
        thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
        offset_x = np.round(inspect_r * np.cos(thetas)).astype(int)
        offset_y = np.round(inspect_r * np.sin(thetas)).astype(int)
        
        # Required points to be considered concave
        threshold_points = round(num_theta * (180 + angle_thresh) / 360.0)

        for corner in corners:
            x, y = corner[0], corner[1]
            
            # Vectorized coordinate generation for the circle
            iter_x = int(round(x)) + offset_x
            iter_y = int(round(y)) + offset_y
            
            # Vectorized Boundary Check
            valid_idx = (iter_x >= 0) & (iter_x < w) & (iter_y >= 0) & (iter_y < h)
            
            valid_x = iter_x[valid_idx]
            valid_y = iter_y[valid_idx]
            
            # Find points inside the CAD mask
            in_mask = mask[valid_y, valid_x] == 1
            num_points_in = np.sum(in_mask)
            
            if num_points_in >= threshold_points:
                # Vectorized shift calculation (Center of mass of the interior points)
                shift_x = np.mean(valid_x[in_mask] - x)
                shift_y = np.mean(valid_y[in_mask] - y)
                
                # Shift corner inwards
                inside_corners.append(np.array([x + shift_x, y + shift_y]))
    
    return inside_corners

def return_cam_position(elevation, azimuth, mesh_path, zoom=1.2):
    
    mesh = pv.read(mesh_path)
    bounds = mesh.bounds
    object_center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
    view_radius = np.linalg.norm([
        bounds[1]-bounds[0],
        bounds[3]-bounds[2],
        bounds[5]-bounds[4]
    ])/2 * zoom

    ## Azimuth and elevation from the image filename
    # elevation, azimuth = extract_el_az_from_viewpath(view_img)
    # Convert spherical coordinates to Cartesian camera position
    ## z-axis in pyvista points is outward from screen (Right hand rule i.e not inward and hence the revolving angle alpha=np.pi/2-azimuth)
    ## elevation is the angle from the +y axis down toward the x-z plane
    ## azimuth is the angle taken from the positive x-axis toward the positive z-axis
    alpha = np.pi/2 - np.radians(azimuth) # angle from z axis towards +x axis in the azimuthal plane y=0
    beta =  np.pi/2 - np.radians(elevation) # angle from x-z plane up towards y axis
    
    camera_position = np.array([object_center[0] + view_radius * np.sin(beta) * np.cos(alpha),
                                object_center[1] + view_radius * np.cos(beta),
                                object_center[2] + view_radius * np.sin(beta) * np.sin(alpha)])

    return camera_position, object_center, view_radius

def mark_spots_in_image(img_path, spot_radius=1, spot_color=(0, 0, 255), spot_positions=[], output_path=None):
    """
    Marks spots on the image at specified positions.

    Parameters:
    - img_path: str, path to the input image.
    - spot_radius: int, radius of the spots to be drawn.
    - spot_color: tuple, BGR color of the spots.
    - spot_positions: list of tuples, each tuple contains (x, y) coordinates for a spot.

    Returns:
    - output_img_path: str, path to the output image with spots marked.
    """
    # Read the image
    img = cv2.imread(img_path)

    # Draw spots on the image
    for pos in spot_positions:
        try:
            if pos.shape[0] !=2: pos = pos.squeeze(0)
        except: pass
        x, y = pos
        x,y = int(round(x)), int(round(y))
        cv2.circle(img, (x,y), spot_radius, spot_color, -1)  # -1 fills the circle
        
    # Save the output image
    if output_path is not None:
        output_img_path = output_path
    elif not img_path.endswith('_with_spots.png'):
        output_img_path = img_path.replace('.png', '_with_spots.png').replace('.jpg', '_with_spots.jpg')
    else:
        output_img_path = img_path
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, img)

    return output_img_path

def generate_or_refine_mesh(
    step_or_mesh_path,
    points_of_interest=[],
    status=None,
    refinement_levels=0,
    h_min=0.5,
    h_max=2.0,
    suffix="",
    out_msh=None,
    verbose=True
):
    points_of_interest = np.asarray(points_of_interest, float)

    # ------------ Output path ------------
    if out_msh is None:
        base, _ = os.path.splitext(step_or_mesh_path)
        out_msh = base.split('~')[0] + f"~{suffix}.msh"
    out_msh = os.path.abspath(out_msh)

    ext = os.path.splitext(step_or_mesh_path)[1].lower()
    is_occ_cad = ext in [".step", ".stp", ".iges", ".igs", ".brep"]
    is_surface_mesh = ext in [".stl", ".obj"]

    gmsh.initialize()

    if verbose:
        gmsh.option.setNumber("General.Terminal", 1)
    else:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)

    # =====================================================================
    # PREVENTATIVE ROBUSTNESS SETTINGS
    # =====================================================================
    gmsh.option.setNumber("Mesh.Algorithm", 6)      # 6 = Frontal-Delaunay for 2D (Highly robust for CAD)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # 10 = HXT for 3D (Fast and robust)
    gmsh.option.setNumber("Mesh.Optimize", 1)       # Optimize elements
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1) # Additional Netgen optimization
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
    
    # Auto-refine tight curves so boundary mesh doesn't self-intersect
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

    if not is_occ_cad and not is_surface_mesh:
        raise ValueError("Input file is not a valid format; cannot create a fresh mesh.")

    if verbose: print(f"[gmsh] Creating initial mesh from {ext}: {step_or_mesh_path}")
    gmsh.open(step_or_mesh_path)

    if is_occ_cad:
        # Heal CAD to prevent inherent overlapping facets
        # gmsh.model.occ.removeAllDuplicates()
        # gmsh.model.occ.synchronize()

        # === CHANGE ADDED HERE: ONLY CONSIDER FIRST VOLUME ===
        vols = gmsh.model.getEntities(dim=3)
        if len(vols) > 1:
            if verbose:
                print(f"[gmsh] Assembly detected ({len(vols)} solids). Keeping only the first solid.")
            # Remove all volumes except the first one. 
            # recursive=True deletes the associated surfaces/lines of the other solids
            gmsh.model.occ.remove(vols[1:], recursive=True)
            gmsh.model.occ.synchronize()
        # =====================================================

        vols = gmsh.model.getEntities(dim=3)
        if len(vols) == 0:
            surfaces = gmsh.model.occ.getEntities(dim=2)
            surf_ids = [s[1] for s in surfaces]
            sl = gmsh.model.occ.addSurfaceLoop(surf_ids)
            gmsh.model.occ.addVolume([sl])
            gmsh.model.occ.synchronize()

    elif is_surface_mesh:
        gmsh.model.mesh.classifySurfaces(
            40 * np.pi / 180,  # angle in radians
            boundary=True,
        )
        gmsh.model.mesh.createGeometry()

    # =====================================================================
    # CASE 1 — CREATE INITIAL MESH FROM CAD FILE
    # =====================================================================
    if status == 'create_initial_mesh':
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_max)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)

    # =====================================================================
    # CASE 2 — LOAD MESH AND DO PROGRESSIVE REFINEMENT
    # =====================================================================
    elif status == 'load_mesh':
        if ext != ".msh":
            raise ValueError("Input file must be a .msh if status=load_mesh")
        if verbose: print(f"[gmsh] Loading existing mesh: {step_or_mesh_path}")
        gmsh.open(step_or_mesh_path)

    # =====================================================================
    # GLOBAL PROGRESSIVE REFINEMENT
    # =====================================================================
    if refinement_levels > 0 and status in ['create_initial_mesh', 'load_mesh']:
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:
            print(f"[gmsh] Warning: Global initial mesh generation failed: {e}")
            
        if verbose or 1:
            print(f"[gmsh] Applying {refinement_levels} global refinement passes...")
        for level in range(refinement_levels):
            if verbose or 1:
                print(f"  - Global refine pass {level+1}/{refinement_levels}")
            gmsh.model.mesh.refine()

    # =====================================================================
    # LOCAL ADAPTIVE REFINEMENT NEAR POINTS OF INTEREST
    # =====================================================================
    else:
        if verbose:
            print(f"[gmsh] Adding local refinement near {len(points_of_interest)} points.")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)
        gmsh.model.mesh.clear()

        pt_tags =[]
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        
        DistMin_base = 2.0 * h_min
        DistMax_base = 4.0 * h_min

        safety = 2 * h_min
        exclude_dist = DistMax_base + safety
        ignored_pois =[]
        
        for p in points_of_interest:
            if len(p) == 0: continue
            y = float(p[1])
            if ymin + exclude_dist < y < ymax - exclude_dist:
                tag = gmsh.model.geo.addPoint(float(p[0]), float(p[1]), float(p[2]))
                pt_tags.append(tag)
            else:
                ignored_pois.append(p)
                
        gmsh.model.geo.synchronize()
        if ignored_pois:
            print(f'Ignored {len(ignored_pois)} POIS: {ignored_pois[:5]} ....')

        # Distance field 
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", pt_tags)

        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

        # -----------------------------------------------------------------
        # ROBUST RETRY LOOP
        # -----------------------------------------------------------------
        max_retries = 20
        success = False
        
        for attempt in range(max_retries):
            try:
                # Progressively relax constraints if it fails
                current_h_min = h_min * (1.2 ** attempt) 
                current_h_max = h_max * (1.2 ** attempt)
                current_DistMin = DistMin_base * (1.5 ** attempt) 
                current_DistMax = DistMax_base * (1.5 ** attempt)

                gmsh.model.mesh.field.setNumber(f_th, "SizeMin", current_h_min)
                gmsh.model.mesh.field.setNumber(f_th, "SizeMax", current_h_max)
                gmsh.model.mesh.field.setNumber(f_th, "DistMin", current_DistMin)
                gmsh.model.mesh.field.setNumber(f_th, "DistMax", current_DistMax)

                if attempt > 0 and verbose:
                    print(f"[gmsh] Retrying mesh generation (Attempt {attempt+1}). Relaxing h_min to {current_h_min:.4f}")

                gmsh.model.mesh.generate(3)
                success = True
                break  # If successful, exit the retry loop
                
            except Exception as e:
                print(f"[gmsh] Mesh generation failed on attempt {attempt+1} due to: {e}")
                gmsh.model.mesh.clear() # Wipe the failed boundary mesh for the next attempt
        
        # Last resort fallback: Remove the points field entirely and just make a basic mesh
        if not success:
            print("[gmsh] ALL RETRIES FAILED. Generating a uniform safety mesh without local refinement to prevent pipeline crash.")
            gmsh.model.mesh.field.setAsBackgroundMesh(0) # Disable the threshold field
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_max)
            gmsh.model.mesh.generate(3)

    # =====================================================================
    # WRITE OUTPUT
    # =====================================================================
    vols = gmsh.model.getEntities(dim=3)
    vol_ids = [v[1] for v in vols]
    gmsh.model.addPhysicalGroup(3, vol_ids, tag=1)
    gmsh.model.setPhysicalName(3, 1, "volume")

    surfaces = gmsh.model.getEntities(dim=2)
    surf_ids = [s[1] for s in surfaces]
    gmsh.model.addPhysicalGroup(2, surf_ids, tag=2)
    gmsh.model.setPhysicalName(2, 2, "boundary")
    
    os.makedirs(os.path.dirname(out_msh), exist_ok=True)
    gmsh.write(out_msh)
    
    if verbose:
        print(f"[gmsh] Wrote final mesh to {out_msh}")
    
    gmsh.finalize()

    return out_msh

def render_tet_mesh_views(
    tet3D_mesh_path,
    output_dir,
    surf_mesh_path,
    n_azimuth=12,
    n_elevation=9,
    window_size=(1000, 1000),
    add_axes=True
):
    """
    Render *wireframe-only* tetrahedral volume mesh.
    All internal edges are visible.
    Uses the same view angles as render_pyvista_views().
    """

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # Load tetrahedral mesh
    # ------------------------
    tet3D_mesh = meshio.read(tet3D_mesh_path)

    # ------------------------
    # Load surf mesh
    # ------------------------
    surf_mesh = trimesh.load(surf_mesh_path)

    object_center = np.mean(surf_mesh.vertices, axis=0)
    zoom = 4.5
    object_radius = np.linalg.norm(surf_mesh.vertices - object_center, axis=1).max() 
    view_radius = zoom * object_radius

    if "tetra" not in tet3D_mesh.cells_dict:
        raise ValueError("Mesh does not contain tetrahedral cells")

    points = tet3D_mesh.points
    tets = tet3D_mesh.cells_dict["tetra"]
    num_cells = tets.shape[0]

    # Build VTK connectivity: [4 v0 v1 v2 v3] repeated
    cells_flat = np.hstack(
        [np.full((num_cells, 1), 4, np.int64), tets]
    ).astype(np.int64).flatten()

    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells_flat, cell_types, points)

    # ------------------------
    # Begin rendering
    # ------------------------
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    # No translucency → no depth peeling
    # plotter.renderer.use_depth_peeling = False
    ## if n_elevation is a list
    if isinstance(n_elevation, list):
        elevations = n_elevation
    else:
        elevations = [-90 + (180 /(n_elevation+1)) * e for e in range(1, n_elevation+1)]

    if isinstance(n_azimuth, list):
        azimuths = n_azimuth
    else:
        azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]

    for elevation in elevations:
        #elevation = -40 + (80 / max(n_elevation - 1, 1)) * e

        for azimuth in azimuths:
            #azimuth = (360 / n_azimuth) * a
            plotter.clear()

            # SAME spherical parametrization
            alpha = np.pi/2 - np.radians(azimuth)
            beta = np.pi/2 - np.radians(elevation)

            cam_x = object_center[0] + view_radius * np.sin(beta) * np.cos(alpha)
            cam_y = object_center[1] + view_radius * np.cos(beta)
            cam_z = object_center[2] + view_radius * np.sin(beta) * np.sin(alpha)

            # ----------------------------------------------------------
            # OPTION-B: Wireframe internal structure
            # ----------------------------------------------------------
            plotter.add_mesh(
                grid,
                style="wireframe",
                color="black",
                line_width=0.4,
                show_edges=True,
            )

            # Axes
            if add_axes:
                plotter.add_axes(
                    interactive=False,
                    line_width=5,
                    labels_off=False,
                    x_color="red",
                    y_color="green",
                    z_color="blue",
                    viewport=(0.0, 0.0, 0.32, 0.35),
                )

            # Set camera
            plotter.camera_position = [(cam_x, cam_y, cam_z), object_center, (0, 1, 0)]

            # Save
            filename = os.path.join(
                output_dir,
                f"view_e{elevation:.0f}_a{azimuth:.0f}.png"
            )
            plotter.render()
            plotter.screenshot(filename)
            

        print(f"✓ Completed elevation {elevation:.1f}°")
        print("Saved:", filename)
        
    plotter.close()
    print("✓ All done.")

def is_contour_in_patch(candidate, patch_num, gridx, gridy, img_shape):
    row, col = (patch_num-1)//gridx, (patch_num-1)%gridx
    patch_x_min = col * (img_shape[1] // gridx)
    patch_x_max = (col + 1) * (img_shape[1] // gridx)
    patch_y_min = row * (img_shape[0] // gridy)
    patch_y_max = (row + 1) * (img_shape[0] // gridy)
    
    # If candidate has a key called 'p_on' which is a list of (x,y) points, check if any of those points are within the patch boundaries
    if isinstance(candidate, dict):
        points = candidate.get('p_on', [])
    else:
        points = candidate

    for pt in points:
        if (patch_x_min <= pt[0] < patch_x_max and patch_y_min <= pt[1] < patch_y_max):
            return True
    return False

def return_img_patch(img_numpy, patch_num, grid_res):
    gridx, gridy = grid_res
    patch_num_to_row_col = {}
    for patch_num in range(1, grid_res[0] * grid_res[1] + 1):
        row, col = (patch_num-1)//grid_res[0], (patch_num-1)%grid_res[0]
        patch_num_to_row_col[patch_num] = (row, col)
    
    if patch_num not in patch_num_to_row_col:
        print(f"Invalid patch number: {patch_num}")
        return None
    row, col = patch_num_to_row_col[patch_num]
    return img_numpy[
        row * (img_numpy.shape[0] // gridy) : (row + 1) * (img_numpy.shape[0] // gridy),
        col * (img_numpy.shape[1] // gridx) : (col + 1) * (img_numpy.shape[1] // gridx)
    ]

def parse_geom_feature_cells(output, prompt_type, infer_views, feature_categories=['I.C.E', 'B.H', 'T.H']):
    pred_cells = {category: [] for category in feature_categories}
    if prompt_type in['geo_max', 'geo_mid']:
        for feature_label in [f"***{cat}: " for cat in feature_categories]:
            cell_strs =[out.split('***')[0] for out in output.split(feature_label)[1:]]
            for cell_str in cell_strs:
                cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
                pred_cells[feature_label[3:-2]].append(cell_list)
    else:
        cell_strs =[out.split('***')[0] for out in output.split('***Cells')[1:]]
        for cell_str in cell_strs:
            cell_list =[int(num.strip()) for num in cell_str.split(',') if num.strip().isdigit()]
            for category in feature_categories:
                pred_cells[category].append(cell_list)

    print("Identified cell numbers:", {cat: [len(lst) for lst in lists] for cat, lists in pred_cells.items()})
    
    # cons_cell_nums = {category: [set() for _ in range(len(infer_views))] for category in feature_categories}
    # for feature_label in feature_categories:
    #     for list_idx, cell_list in enumerate(pred_cells[feature_label]):
    #         if list_idx >= len(infer_views):
    #             print(f"Warning: LLM returned extra predictions for {feature_label}. Ignoring extras.")
    #             break
    #         for cell_num in cell_list:
    #             for look_x in range(-cons_cell_lookup, cons_cell_lookup +1):
    #                 for look_y in range(-cons_cell_lookup, cons_cell_lookup +1):
    #                     neighbor_cell_num = cell_num + look_x + look_y * grid_res[0]
    #                     if 1 <= neighbor_cell_num <= grid_res[0] * grid_res[1]:
    #                         cons_cell_nums[feature_label][list_idx].add(neighbor_cell_num)

    #{'I.C.E':[[],[],... ], 'B.H': [[],[],..], 'T.H': [[],[],..]} -> [{'I.C.E': [...], 'B.H': [...], 'T.H': [...]}, {'I.C.E': [...], 'B.H': [...], 'T.H': [...]}, ...]}
    pred_cells_list = []
    num_views = len(infer_views)
    for view_idx in range(num_views):
        view_cells = {category: pred_cells[category][view_idx] if view_idx < len(pred_cells[category]) else [] for category in feature_categories}
        pred_cells_list.append(view_cells)

    return pred_cells_list

# def extract_all_contours_geometry(img, min_contour_len=30, sampling_step=15, pixel_offset=5):
#     """
#     Extracts (p_on, p_in, p_out) triplets for ALL valid contours in the entire image.
#     The number of samples is dynamically scaled based on the contour's length.
    
#     Args:
#         img: The full input image (BGR or Grayscale).
#         min_contour_len: Minimum perimeter length to be considered a valid feature.
#         sampling_step: A sample is taken roughly every 'sampling_step' pixels along the contour.
#         pixel_offset: Distance in pixels for p_in and p_out from p_on.
        
#     Returns:
#         List of dictionaries, where each dictionary contains geometry data for one contour.
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Use RETR_LIST to get all contours (both outer boundaries and inner holes)
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
#     all_geometries = []
    
#     for c in contours:
#         if len(c) < min_contour_len:
#             continue
            
#         # 1. Dynamic Sampling: Scale number of samples with contour length
#         num_samples = max(5, len(c) // sampling_step)
        
#         # 2. Find Centroid: Used to consistently point 'p_in' towards the interior
#         M = cv2.moments(c)
#         if M['m00'] != 0:
#             cx = M['m10'] / M['m00']
#             cy = M['m01'] / M['m00']
#             centroid = np.array([cx, cy])
#         else:
#             # Fallback for degenerate contours
#             centroid = np.mean(c[:, 0, :], axis=0)
            
#         # 3. Determine if straight (A line has ~0 area compared to its length)
#         area = cv2.contourArea(c)
#         length = cv2.arcLength(c, closed=True)
#         is_straight = area < (length * 1.5) # Threshold for "straightness/flatness"
        
#         points_on, points_in, points_out = [], [], []
        
#         # Sample points evenly across the contour
#         indices = np.linspace(0, len(c) - 1, num_samples, dtype=int)
        
#         for idx in indices:
#             p_on = c[idx][0]
            
#             # Calculate local tangent and normal (handle wrap-around for closed contours)
#             idx_prev = (idx - 3) % len(c)
#             idx_next = (idx + 3) % len(c)
#             pt1, pt2 = c[idx_prev][0], c[idx_next][0]
            
#             dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
#             tangent_len = math.hypot(dx, dy)
#             if tangent_len == 0: continue
            
#             nx, ny = -dy / tangent_len, dx / tangent_len
#             normal = np.array([nx, ny])
            
#             # Align local normal to point towards the centroid (defining the "inside")
#             vec_to_centroid = centroid - p_on
#             if np.dot(normal, vec_to_centroid) > 0:
#                 dir_in = normal
#             else:
#                 dir_in = -normal
                
#             p_in = p_on + dir_in * pixel_offset
#             p_out = p_on - dir_in * pixel_offset
            
#             points_on.append(p_on)
#             points_in.append(p_in)
#             points_out.append(p_out)

#         if not points_on:
#             continue

#         all_geometries.append({
#             "p_on": np.array(points_on),   
#             "p_in": np.array(points_in),   
#             "p_out": np.array(points_out), 
#             "is_straight": is_straight,
#             "contour_ref": c  # Keep a reference to the original contour for plotting
#         })

#     return all_geometries

# def extract_all_contours_geometry(img, min_contour_len=30, sampling_step=15, pixel_offset=5, output_folder='./'):
#     """
#     Extracts geometry triplets and generates a separate visualization image for each contour.
#     """
#     if output_folder and not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Prepare image
#     display_base = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
#     edges = cv2.Canny(gray, 50, 150)
    
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
#     all_geometries = []
    
#     for i, c in enumerate(contours):
#         if len(c) < min_contour_len:
#             continue
            
#         # --- 1. Geometry Calculation Logic ---
#         num_samples = max(5, len(c) // sampling_step)
#         M = cv2.moments(c)
#         if M['m00'] != 0:
#             centroid = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
#         else:
#             centroid = np.mean(c[:, 0, :], axis=0)
            
#         area = cv2.contourArea(c)
#         length = cv2.arcLength(c, closed=True)
#         is_straight = area < (length * 1.2) # Tightened threshold
        
#         points_on, points_in, points_out = [], [], []
#         indices = np.linspace(0, len(c) - 1, num_samples, dtype=int)
        
#         for idx in indices:
#             p_on = c[idx][0].astype(float)
            
#             idx_prev = (idx - 3) % len(c)
#             idx_next = (idx + 3) % len(c)
#             pt1, pt2 = c[idx_prev][0].astype(float), c[idx_next][0].astype(float)
            
#             dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
#             tangent_len = math.hypot(dx, dy)
#             if tangent_len == 0: continue
            
#             # Normal calculation
#             nx, ny = -dy / tangent_len, dx / tangent_len
#             normal = np.array([nx, ny])
            
#             # Align normal to centroid
#             vec_to_centroid = centroid - p_on
#             dir_in = normal if np.dot(normal, vec_to_centroid) > 0 else -normal
                
#             p_in = p_on + dir_in * pixel_offset
#             p_out = p_on - dir_in * pixel_offset
            
#             points_on.append(p_on)
#             points_in.append(p_in)
#             points_out.append(p_out)

#         if not points_on:
#             continue

#         # --- 2. Create Individual Visualization Image ---
#         # We create a black image and draw ONLY this contour and its geometry
#         individual_img = np.zeros_like(display_base)
        
#         # Draw the full contour line in white
#         cv2.drawContours(individual_img, [c], -1, (255, 255, 255), 1)
        
#         # Draw the Triplets: p_on (Yellow), p_in (Green), p_out (Red)
#         for on, inp, out in zip(points_on, points_in, points_out):
#             # Draw lines connecting the triplet to visualize the normal
#             cv2.line(individual_img, tuple(inp.astype(int)), tuple(out.astype(int)), (100, 100, 100), 1)
#             # Draw points
#             cv2.circle(individual_img, tuple(on.astype(int)), 2, (0, 255, 255), -1) # Yellow
#             cv2.circle(individual_img, tuple(inp.astype(int)), 2, (0, 255, 0), -1)   # Green
#             cv2.circle(individual_img, tuple(out.astype(int)), 2, (0, 0, 255), -1)   # Red

#         # Optional: Add text label
#         label = "Curve" if not is_straight else "Line"
#         cv2.putText(individual_img, f"ID: {i} | {label}", (20, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#         # --- 3. Save Image if path provided ---
#         if output_folder:
#             file_name = f"contour_{i}_{label.lower()}.png"
#             cv2.imwrite(os.path.join(output_folder, file_name), individual_img)

#         all_geometries.append({
#             "id": i,
#             "p_on": np.array(points_on),   
#             "p_in": np.array(points_in),   
#             "p_out": np.array(points_out), 
#             "is_straight": is_straight,
#             "visual": file_name # Returning the image object as well
#         })

#     return all_geometries

import cv2
import numpy as np
import math
import os

def extract_all_contours_geometry(img, min_contour_len=30, sampling_step=15, pixel_offset=5, 
            cam_pos=None, F_pos=None, up_cam_vec=None, img_dims=None, 
            parallel_scale=None, mesh=None):
    
    ## For debug only 
    # output_folder = './'
    # if output_folder and not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    display_base = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    all_geometries = []
    contour_point_counts = [] # To track how many points each contour has

    # --- Phase 1: Geometry Extraction ---
    for i, c in enumerate(contours):
        if len(c) < min_contour_len:
            continue
            
        area = cv2.contourArea(c)
        length = cv2.arcLength(c, closed=True)
        is_straight = area < (length * 1.2)
        
        points_on, points_in, points_out = [], [], []
        num_samples = max(5, len(c) // sampling_step)
        indices = np.linspace(0, len(c) - 1, num_samples, dtype=int)
        
        for idx in indices:
            p_on = c[idx][0].astype(float)
            idx_prev = (idx - 5) % len(c)
            idx_next = (idx + 5) % len(c)
            pt1, pt2 = c[idx_prev][0].astype(float), c[idx_next][0].astype(float)
            
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            tangent_len = math.hypot(dx, dy)
            if tangent_len == 0: continue
            
            nx, ny = -dy / tangent_len, dx / tangent_len
            normal = np.array([nx, ny])
            
            # Point-in-polygon test for 'in' direction
            test_pt = p_on + normal * 1.0 
            dist = cv2.pointPolygonTest(c, (float(test_pt[0]), float(test_pt[1])), True)
            dir_in = normal if dist >= 0 else -normal

            p_in = p_on + dir_in * pixel_offset
            p_out = p_on - dir_in * pixel_offset
            
            # Handle tight curves
            if cv2.pointPolygonTest(c, (float(p_in[0]), float(p_in[1])), False) < 0:
                p_in = p_on + dir_in * (max(0, dist))

            points_on.append(p_on)
            points_in.append(p_in)
            points_out.append(p_out)

        if not points_on:
            continue

        # Visualization
        individual_img = np.zeros_like(display_base)
        cv2.drawContours(individual_img, [c], -1, (255, 255, 255), 1)
        for on, inp, out in zip(points_on, points_in, points_out):
            cv2.line(individual_img, tuple(inp.astype(int)), tuple(out.astype(int)), (100, 100, 100), 1)
            cv2.circle(individual_img, tuple(on.astype(int)), 2, (0, 255, 255), -1) 
            cv2.circle(individual_img, tuple(inp.astype(int)), 2, (0, 255, 0), -1)   
            cv2.circle(individual_img, tuple(out.astype(int)), 2, (0, 0, 255), -1)   

        label = "curve" if not is_straight else "line"
        # file_name = f"contour_{i}_{label}.png"
        # cv2.imwrite(os.path.join(output_folder, file_name), individual_img)

        # Store basic geometry
        all_geometries.append({
            "id": i,
            "p_on": np.array(points_on),   
            "p_in": np.array(points_in),   
            "p_out": np.array(points_out), 
            "is_straight": is_straight,
            #"visual": file_name
        })
        contour_point_counts.append(len(points_on))

    if not all_geometries:
        return []

    # --- Phase 2: Vectorized Raycasting ---
    # Stack all points into one master list for efficient processing
    # Order: [All p_ins, All p_ons, All p_outs]
    all_p_in = np.vstack([g['p_in'] for g in all_geometries])
    all_p_on = np.vstack([g['p_on'] for g in all_geometries])
    all_p_out = np.vstack([g['p_out'] for g in all_geometries])
    
    master_pixels = np.vstack([all_p_in, all_p_on, all_p_out])
    total_samples = all_p_in.shape[0]

    all_hits_3d = pixel_to_mesh(
        cam_pos=cam_pos, F_pos=F_pos, u=master_pixels[:, 0], v=master_pixels[:, 1],
        up_cam_vec=up_cam_vec, img_H=img_dims[0], img_W=img_dims[1],
        parallel_scale=parallel_scale, mesh=mesh, orthographic=True, 
        return_no_location=True
    )

    # Compute depths
    all_depths = np.array([
        np.linalg.norm(hit - cam_pos) if hit is not None else np.inf 
        for hit in all_hits_3d
    ])

    # Split the master depth array into in, on, and out blocks
    d_ins_master = all_depths[:total_samples]
    d_ons_master = all_depths[total_samples : 2 * total_samples]
    d_outs_master = all_depths[2 * total_samples:]

    # --- Phase 3: Mapping depths back to specific contours ---
    current_idx = 0
    for idx, count in enumerate(contour_point_counts):
        # Slice out the segments belonging to this specific contour
        all_geometries[idx]['d_in']  = d_ins_master[current_idx : current_idx + count]
        all_geometries[idx]['d_on']  = d_ons_master[current_idx : current_idx + count]
        all_geometries[idx]['d_out'] = d_outs_master[current_idx : current_idx + count]
        
        current_idx += count

    return all_geometries

def detect_extrusion_boundary_or_protruding_boundary_vectorized(
    geom, tol=0.5, inf_threshold=1e5, consensus_ratio=0.3
):
    """
    Detects if a contour represents a geometric boundary directly using precomputed depths.
    """
    num_samples = len(geom['p_on'])
    if num_samples == 0:
        return False, {key: [] for key in['p_on', 'p_in', 'p_out']}

    # Directly load the precomputed depths
    d_ins = np.array(geom['d_in'])
    d_ons = np.array(geom['d_on'])
    d_outs = np.array(geom['d_out'])

    # 1. Identify which rays successfully hit the mesh vs missed (void)
    hit_in  = d_ins < inf_threshold
    hit_on  = d_ons < inf_threshold
    hit_out = d_outs < inf_threshold

    # 2. INTERNAL GEOMETRY CONDITIONS: All three rays hit the mesh
    all_hit = hit_in & hit_on & hit_out
    
    # Safely compute absolute differences only where all rays hit
    diff_in_on = np.zeros(num_samples)
    diff_out_on = np.zeros(num_samples)
    diff_in_on[all_hit] = np.abs(d_ins[all_hit] - d_ons[all_hit])
    diff_out_on[all_hit] = np.abs(d_outs[all_hit] - d_ons[all_hit])

    cond_step_in  = all_hit & (diff_in_on <= tol)  & (d_ins > d_outs + tol)  
    cond_step_out = all_hit & (diff_out_on <= tol) & (d_ins < d_outs - tol)
    cond_valley   = all_hit & (d_ons > d_ins + tol) & (d_ons > d_outs + tol)            

    # 3. Combine all successful conditions into one global mask
    success_mask = cond_step_in | cond_step_out | cond_valley
    
    # Calculate ratio and extract only the partial geometry that satisfies the boundaries
    ratio = np.sum(success_mask) / num_samples
    
    # Ensure arrays before masking just in case they are standard lists
    ec_pc_partial = {key: np.array(geom[key])[success_mask] for key in ['p_on', 'p_in', 'p_out', 'd_on' ,'d_in', 'd_out']}
    
    return ratio >= consensus_ratio, ec_pc_partial


def detect_through_hole(geom, inf_threshold=1e5, consensus_ratio=0.4):
    """
    Detects a through hole if the inside depth goes to infinity but the outside hits the object.
    """
    # If the feature is just a straight line, it cannot be a hole
    if geom.get('is_straight', False):
        return False 

    num_samples = len(geom['p_in'])
    if num_samples == 0:
        return False

    # Directly use precomputed depth
    d_ins = np.array(geom['d_in'])
    d_outs = np.array(geom['d_out'])

    # Success: inside hits infinity, outside hits mesh
    success_mask = (d_ins > inf_threshold) & (d_outs < inf_threshold)
    
    ratio = np.sum(success_mask) / num_samples
    
    return ratio >= consensus_ratio


def is_concave(contour_coords, inf_threshold=1e4):
    """
    Checks if a contour patch is concave based on precomputed depths.
    A contour is concave if the inside of its curve points towards empty space (depth > inf_threshold).
    """
    p_on = np.array(contour_coords['p_on'])
    p_in = np.array(contour_coords['p_in'])
    p_out = np.array(contour_coords['p_out'])
    
    # Depths are now guaranteed
    d_in = np.array(contour_coords['d_in'])
    d_out = np.array(contour_coords['d_out'])

    n = len(p_on)
    if n < 3:
        return False
        
    step = max(1, n // 10) 
    concave_votes = 0
    total_valid_points = 0
    
    for i in range(step, n - step):
        pt = p_on[i]
        
        # BEND VECTOR: Mathematically points towards the "center" of the curve
        pt_prev = p_on[i - step]
        pt_next = p_on[i + step]
        bend_vec = (pt_prev - pt) + (pt_next - pt) 
        
        # LOCAL NORMALS
        vec_in = p_in[i] - pt
        vec_out = p_out[i] - pt
        
        # DOT PRODUCTS
        dot_in = np.dot(bend_vec, vec_in)
        dot_out = np.dot(bend_vec, vec_out)
        
        is_pt_concave = False
        
        # Check depth at the center of the curve
        if dot_out > dot_in and dot_out > 0:
            if d_out[i] > inf_threshold:
                is_pt_concave = True
        elif dot_in > dot_out and dot_in > 0:
            if d_in[i] > inf_threshold:
                is_pt_concave = True
                
        if is_pt_concave:
            concave_votes += 1
            
        total_valid_points += 1
        
    if total_valid_points == 0:
        return False
        
    return concave_votes > (total_valid_points * 0.5)


def detect_parts_of_contour(contours, cell_list, gridx, gridy, img_shape):
    """
    Extracts parts of the global contours that intersect with specific patches 
    and determines their concavity.
    """
    pixel_coords = []
    concave_bools =[] 
    
    for contour in contours:
        # We now explicitly initialize d_in and d_out containers
        contour_coords = {'p_on':[], 'p_in': [], 'p_out': [], 'd_in': [], 'd_out':[]}
        
        for cell_num in cell_list:
            for pt_id, pt in enumerate(contour['p_on']):
                if is_contour_in_patch({'p_on': [pt]}, cell_num, gridx, gridy, img_shape):
                    contour_coords['p_on'].append(pt)
                    contour_coords['p_in'].append(contour['p_in'][pt_id])
                    contour_coords['p_out'].append(contour['p_out'][pt_id])
                    
                    # Store depth data alongside the coordinates
                    contour_coords['d_in'].append(contour['d_in'][pt_id])
                    contour_coords['d_out'].append(contour['d_out'][pt_id])
        
        # Only evaluate if we actually found points in the patch
        if len(contour_coords['p_on']) > 0:
            pixel_coords.append(contour_coords)
            
            # Check if the contour coords are concave
            is_conc = is_concave(contour_coords)
            concave_bools.append(is_conc)

    return pixel_coords, concave_bools
    
# import numpy as np

def smooth_contour_coords(coords, window_size=5):
    """Applies a 1D moving average to smooth out pixel-staircase noise."""
    if len(coords) < window_size:
        return coords
    pad_w = window_size // 2
    # Pad the edges to maintain array length
    padded_x = np.pad(coords[:, 0], (pad_w, pad_w), mode='edge')
    padded_y = np.pad(coords[:, 1], (pad_w, pad_w), mode='edge')
    
    smooth_x = np.convolve(padded_x, np.ones(window_size)/window_size, mode='valid')
    smooth_y = np.convolve(padded_y, np.ones(window_size)/window_size, mode='valid')
    return np.column_stack((smooth_x, smooth_y))

def extract_concave_parts(contours, depth_map=None, depth_tolerance=0.5, min_bend=0.5):
    """
    Extracts purely concave segments by checking if the curve bends around empty space.
    
    Args:
        depth_tolerance: Minimum depth difference to be considered "background/void". 
                         (e.g., if surface is at depth 5.0, background must be > 5.5).
        min_bend: Minimum curvature magnitude. (Filters out flat lines).
    """
    if depth_map is not None:
        H, W = depth_map.shape[:2]
        
    all_concave_parts =[]
    
    for idx, contour in enumerate(contours):
        p_on = np.array(contour['p_on'])
        p_in = np.array(contour['p_in'])
        p_out = np.array(contour['p_out'])
        n = len(p_on)
        
        # Cannot accurately determine curvature on tiny fragments
        if n < 10: 
            continue
            
        # 1. PRE-SMOOTHING: Destroy Canny pixel-staircase aliasing
        # This makes the math treat it like a smooth vector curve rather than a grid of blocks
        p_on_smooth = smooth_contour_coords(p_on, window_size=5)
            
        # Dynamically determine lookahead step based on size
        step = max(4, n // 20) 
        
        is_concave_mask = np.zeros(n, dtype=bool)
        
        d_in_arr = contour.get('d_in', None)
        d_out_arr = contour.get('d_out', None)
        d_on_arr = contour.get('d_on', None) # Need d_on to check relative depth
        
        # 2. DROP MODULO: Iterate ONLY through the safe inner bounds
        # This completely prevents open curves from wrapping around and creating fake bends
        for i in range(step, n - step):
            pt = p_on_smooth[i]
            
            # Center of Curvature Vector
            pt_prev = p_on_smooth[i - step]
            pt_next = p_on_smooth[i + step]
            bend_vec = (pt_prev - pt) + (pt_next - pt)
            bend_mag = np.linalg.norm(bend_vec)
            
            # If the segment is practically a straight line, skip it
            if bend_mag < min_bend:
                continue
                
            # Normalize vectors
            bend_unit = bend_vec / bend_mag
            vec_in = (p_in[i] - p_on[i])
            vec_out = (p_out[i] - p_on[i])
            
            vec_in_unit = vec_in / (np.linalg.norm(vec_in) + 1e-6)
            vec_out_unit = vec_out / (np.linalg.norm(vec_out) + 1e-6)
            
            # Dot products check which way the belly of the curve points
            dot_in = np.dot(bend_unit, vec_in_unit)
            dot_out = np.dot(bend_unit, vec_out_unit)
            
            # FETCH DEPTH
            if d_in_arr is not None and d_on_arr is not None:
                depth_in, depth_out, depth_on = d_in_arr[i], d_out_arr[i], d_on_arr[i]
            else:
                v_in, u_in = np.clip(int(p_in[i][1]), 0, H-1), np.clip(int(p_in[i][0]), 0, W-1)
                v_out, u_out = np.clip(int(p_out[i][1]), 0, H-1), np.clip(int(p_out[i][0]), 0, W-1)
                v_on, u_on = np.clip(int(p_on[i][1]), 0, H-1), np.clip(int(p_on[i][0]), 0, W-1)
                
                depth_in = depth_map[v_in, u_in]
                depth_out = depth_map[v_out, u_out]
                depth_on = depth_map[v_on, u_on]

            # 3. RELATIVE DEPTH CONCAVITY LOGIC
            # A curve is concave if it bends towards a space that is DEEPER than the object itself.
            # dot > 0.8 ensures the bend is strictly aligned with the normal (not a diagonal glitch)
            
            if dot_out > 0.8 and (depth_out > depth_on + depth_tolerance):
                is_concave_mask[i] = True
            elif dot_in > 0.8 and (depth_in > depth_on + depth_tolerance):
                is_concave_mask[i] = True
                    
        # 4. DENOISING: Require continuous agreement
        # Remove random 1-pixel anomalies; requires a small neighborhood to agree
        final_concave_pixels =[]
        for i in range(step, n - step):
            if is_concave_mask[i]:
                # Check neighbors
                window = is_concave_mask[i-2 : i+3]
                if np.sum(window) >= 3:
                    final_concave_pixels.append(p_on[i]) # Append original point, not smoothed
                
        if len(final_concave_pixels) > 0:
            all_concave_parts.append({
                'contour_id': idx,
                'concave_pixels': np.array(final_concave_pixels)
            })
            
    return all_concave_parts

# def extract_concave_parts(contours, depth_map=None, inf_threshold=1e4):
#     """
#     Analyzes every point on all contours to extract purely the concave segments.
#     A segment is concave if the contour wraps around empty space (depth > inf_threshold).
    
#     Args:
#         contours: List of contour geometries from `extract_all_contours_geometry`.
#         depth_map: Dense 2D numpy array of shape (H, W) containing depth at every pixel. 
#                    (Optional if 'd_in' and 'd_out' are already present in the contour dicts).
#         inf_threshold: Depth value threshold marking "empty space" or background void.
        
#     Returns:
#         List of dictionaries containing the extracted concave pixels for each contour.
#     """
#     if depth_map is not None:
#         H, W = depth_map.shape[:2]
        
#     all_concave_parts =[]
    
#     for idx, contour in enumerate(contours):
#         p_on = np.array(contour['p_on'])
#         p_in = np.array(contour['p_in'])
#         p_out = np.array(contour['p_out'])
        
#         n = len(p_on)
#         # Cannot compute curvature with fewer than 3 points
#         if n < 3:
#             continue
            
#         # Dynamically determine step size to avoid pixel-level aliasing (staircase effect)
#         # Using a broader window ensures a smooth mathematical curve vector
#         step = max(2, n // 20) 
        
#         concave_pixels =[]
        
#         # Check if depth arrays are already embedded in the contour dict
#         d_in_arr = contour.get('d_in', None)
#         d_out_arr = contour.get('d_out', None)
        
#         # Evaluate concavity point-by-point
#         for i in range(n):
#             pt = p_on[i]
            
#             # 1. BEND VECTOR: Compute center of curvature using wrap-around indexing
#             pt_prev = p_on[(i - step) % n]
#             pt_next = p_on[(i + step) % n]
#             bend_vec = (pt_prev - pt) + (pt_next - pt)
            
#             # 2. LOCAL NORMALS: Vectors pointing 'in' and 'out'
#             vec_in = p_in[i] - pt
#             vec_out = p_out[i] - pt
            
#             # 3. DOT PRODUCT: Does it bend towards 'in' or 'out'?
#             dot_in = np.dot(bend_vec, vec_in)
#             dot_out = np.dot(bend_vec, vec_out)
            
#             # 4. FETCH LOCAL DEPTH
#             if d_in_arr is not None and d_out_arr is not None:
#                 depth_in = d_in_arr[i]
#                 depth_out = d_out_arr[i]
#             elif depth_map is not None:
#                 # Map coordinates to integers for dense depth_map indexing
#                 u_in, v_in = int(np.round(p_in[i][0])), int(np.round(p_in[i][1]))
#                 u_out, v_out = int(np.round(p_out[i][0])), int(np.round(p_out[i][1]))
                
#                 # Safe bounding to prevent out-of-bounds errors on image edges
#                 v_in, u_in = np.clip(v_in, 0, H-1), np.clip(u_in, 0, W-1)
#                 v_out, u_out = np.clip(v_out, 0, H-1), np.clip(u_out, 0, W-1)
                
#                 depth_in = depth_map[v_in, u_in]
#                 depth_out = depth_map[v_out, u_out]
#             else:
#                 raise ValueError("Must provide either depth_map or precomputed d_in/d_out arrays.")
                
#             # 5. CONCAVITY TEST
#             is_pt_concave = False
            
#             if dot_out > dot_in and dot_out > 0:
#                 # The curve bends towards 'out'. Is 'out' empty space?
#                 if depth_out > inf_threshold:
#                     is_pt_concave = True
#             elif dot_in > dot_out and dot_in > 0:
#                 # The curve bends towards 'in'. Is 'in' empty space?
#                 if depth_in > inf_threshold:
#                     is_pt_concave = True
                    
#             if is_pt_concave:
#                 concave_pixels.append(pt)
                
#         # If we found concave pixels in this contour, store them
#         if len(concave_pixels) > 0:
#             all_concave_parts.append({
#                 'contour_id': idx, # So you know which original contour this belongs to
#                 'concave_pixels': np.array(concave_pixels)
#             })
            
#     return all_concave_parts

# --- MOCK DATA FOR DEMONSTRATION ---
if __name__ == '__main__':
    # You would replace this with your actual mesh and camera setup
    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_7.04.2026'
    cad_object = '00210097'
    load_case = 'torsion'
    num_views = 3
    llm_name = "google/gemini-3-flash-preview" #"openai/gpt-5-mini"
    
    mesh_file_path = f"{parent_dir}/{cad_object}/renders_pyvista/{cad_object}.obj"
    mesh = trimesh.load(mesh_file_path)
    
    view_dir = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial'
    all_loaded_view_paths =[f'{parent_dir}/{cad_object}/renders_pyvista_mesh_{load_case}/{view_name}' for view_name in os.listdir(view_dir) if not view_name.startswith('view_e-90') and not view_name.startswith('view_e90')]
    perspective_views = np.random.choice(all_loaded_view_paths, size=2, replace=False).tolist()

    all_orthoview_paths = [f'{view_dir}/{view_name}' for view_name in sorted(os.listdir(view_dir))]
    infer_views_paths =[]
    with open(f'{parent_dir}/{cad_object}/pred_ortho_views2.txt', 'r') as fread:
        # Read lines, strip whitespace
        raw_lines =[line.strip() for line in fread.readlines() if line.strip()][:num_views]
        
        for line in raw_lines:
            # Extract JUST the image name (e.g., 'view_e0_a0.png') from the old absolute path
            view_name = os.path.basename(line)
            
            # Reconstruct the correct path using the CURRENT parent_dir
            current_correct_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{view_name}'
            
            # Make sure the file actually exists before we try to process it
            if not os.path.exists(current_correct_path):
                raise FileNotFoundError(f"Image missing or network drive timeout: {current_correct_path}")
            
            infer_views_paths.append(current_correct_path)

    #render_mesh_views(mesh_file_path, n_azimuth=[0], n_elevation=[-90, 90], orthographic=True, output_dir=view_dir, add_axes=False)
    #render_mesh_views_with_load(mesh_file_path, n_azimuth=12, n_elevation=9, orthographic=True, output_dir_prefix=f'{parent_dir}/{cad_object}/renders_pyvista_mesh', add_axes=True, loading_type=load_case)
    
    infer_views_paths.extend([f'{view_dir}/{view_name}' for view_name in os.listdir(view_dir) if view_name.startswith('view_e-90_a0') or view_name.startswith('view_e90_a0')])
    #remaining_views = list(set(all_loaded_view_paths) - set(perspective_views))
    #infer_views_paths = np.random.choice(remaining_views, size=2, replace=False).tolist()

    #infer_views =[os.path.basename(view_path) for view_path in infer_views_paths]
    infer_views_paths[0] = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/view_e0_a180.png'
    experiment_name = f'try_CV_detection_{load_case}'
    
    output_grid_dir = f'{parent_dir}/{cad_object}/{experiment_name}/renders_pyvista_mesh_initial'
    os.makedirs(output_grid_dir, exist_ok=True)
    
    gridded_views =[]
    gridx, gridy = 11, 11
    for img_path in tqdm(infer_views_paths, desc="Overlaying Grids", file=sys.stdout):
        gridded_path = overlay_grid(img_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
                                    font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
                                    line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    arrow_cell=None, output_dir=output_grid_dir)
        gridded_views.append(gridded_path)

    print(f'Top Views Gridded: {gridded_views}')

    # --- Prompt Construction ---
    prompt = get_prompt_1(load_case, prompt_type='geomax', num_perspective_views=len(perspective_views), num_gridded_views=len(gridded_views)-2)  # You would define this function to return your actual prompt text

    messages =[]
    inference_imgs =[]
    img_paths = perspective_views + gridded_views
    for img_path in img_paths:
        with open(img_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            inference_imgs.append(encoded_img)

    content =[{"type": "text", "text": prompt}]
    content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
    messages.append({"role": "user", "content": content})
    
    # --- Protected API Call ---
    print(f"Prompt Stage 1 Region Selection: Waiting for OpenRouter API Response ({llm_name})...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": llm_name, "temperature": 0.0, "messages": messages},
            timeout=60  # Prevent infinite hanging
        )
        response.raise_for_status() # Catches 429, 502, etc.
        response_data = response.json()
        
        if 'choices' not in response_data:
            print(f"API Error Response format: {response_data}")
            sys.exit(1)
            
        output = response_data['choices'][0]['message']['content']
        print("Response Received:", output)
        
    except Exception as e:
        print(f"API Request Failed: {e}")
        sys.exit(1) # Exit cleanly to inform parent script

    pred_cells_lists = parse_geom_feature_cells(output, prompt_type='geomax', infer_views=infer_views_paths, feature_categories=['I.C.E', 'E.C/P.C', 'T.H', 'F'])
    
    output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}'
    os.makedirs(output_2d_dir, exist_ok=True)
    
    pixel_coords_ICE, pixel_coords_EC_PC, pixel_coords_TH, pixel_coords_F = [], [], [], []
    output_img_paths = set()  # Use a set to avoid duplicates
    
    ## Define point density for point prediction and projection, will use the same for the geometric heuristic
    bbox_min, bbox_max = mesh.bounds  # shape (2, 3)
    point_density = np.linalg.norm(bbox_max - bbox_min) / 66.667  # Adjust divisor for more/less density
    
    for i, view_path in enumerate(gridded_views):
        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view_path)}'
        
        out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8
        up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,1)
        img = cv2.imread(view_path)
        if img is None:
            print(f"Failed to load image: {view_path}")
            continue
        
        s_step, p_offset = calculate_dynamic_sampling_params(
        point_density=point_density, 
        img_shape=(1000, 1000), 
        cam_pos=cam_pos, 
        F_pos=F_pos, 
        fov_deg=75, 
        parallel_scale=parallel_scale, 
        orthographic=True)

        img_numpy = np.array(img)
        geom = extract_all_contours_geometry(img_numpy, sampling_step=s_step, pixel_offset=p_offset)
        ec_pc_candidates, th_candidates =[], []
        for contour in geom:
            # insert sampling distance into each
            if detect_extrusion_boundary_or_protruding_boundary_vectorized(geom=contour, mesh=mesh, cam_pos=cam_pos, F_pos=F_pos, parallel_scale=parallel_scale, up_cam_vec=up_cam_vec):  # You would fill in the actual mesh and camera parameters here
                ec_pc_candidates.append(contour)
            elif detect_through_hole(geom=contour, mesh=mesh, cam_pos=cam_pos, F_pos=F_pos, parallel_scale=parallel_scale, up_cam_vec=up_cam_vec):  # You would fill in the actual mesh and camera parameters here
                th_candidates.append(contour)
        
        # Plot ec_pc candidates for visual debugging
        debug_img = img.copy()
        for candidate in ec_pc_candidates:
            for pt in candidate['p_on']:
                cv2.circle(debug_img, tuple(pt.astype(int)), radius=3, color=(255, 0, 0), thickness=-1)
        debug_out_path = f'{output_2d_dir}/debug_ec_pc_{os.path.basename(view_path)}'
        cv2.imwrite(debug_out_path, debug_img)
        print(f"Saved EC/PC candidate debug image to {debug_out_path}") 

        #Plot th candidates for visual debugging
        debug_img_th = img.copy()
        for candidate in th_candidates:
            for pt in candidate['p_on']:
                cv2.circle(debug_img_th, tuple(pt.astype(int)), radius=3, color=(0, 0, 255), thickness=-1)
        debug_out_path_th = f'{output_2d_dir}/debug_th_{os.path.basename(view_path)}'
        cv2.imwrite(debug_out_path_th, debug_img_th)
        print(f"Saved TH candidate debug image to {debug_out_path_th}")

        # Here you would call analyze_cell_patch_vectorized() for each cell patch in the image
        pred_cells = pred_cells_lists[i]

        pixels_coords_view =[]    
        for category, patch_nums in pred_cells.items():
            
            
            if category == 'I.C.E':
                print(f"Analyzing Internal Concave Edges in {view_path} for patches {patch_nums}")
                pixel_candidates = []
                for patch_num in patch_nums:
                    ## convert image to numpy array and extract the patch corresponding to patch_num
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    candidates = edge_detection_contour(view_path, patch_num, gridx, gridy, angle_thresh=20)
                    pixel_candidates.extend(candidates)
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 255, 0), 
                                                spot_positions=pixel_candidates, output_path=out_path))
                pixel_coords_ICE.append(pixel_candidates)
    
            elif category == 'E.C/P.C':
                print(f"Analyzing Extruded Contour / Portruding Contour in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for extrusion boundaries / protruding boundaries
                pixel_candidates = []
                for patch_num in patch_nums:
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    for candidate in ec_pc_candidates:
                        
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixel_candidates.extend(candidate['p_on'])
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                                                spot_positions=pixel_candidates, output_path=out_path))
                pixel_coords_EC_PC.append(pixel_candidates)

            elif category == 'T.H':
                print(f"Analyzing Through Holes / Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for through holes / fillets
                pixel_candidates = []
                for patch_num in patch_nums:
                    patch = return_img_patch(img_numpy, patch_num, grid_res=(gridx, gridy))
                    for candidate in th_candidates:
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixel_candidates.extend(candidate['p_on'])
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 0, 255), 
                                                spot_positions=pixel_candidates, output_path=out_path))
                pixel_coords_TH.append(pixel_candidates)
            
            elif category == 'F':
                print(f"Analyzing Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for fillets
            
                pixel_candidates = detect_parts_of_contour(geom, patch_nums, gridx, gridy, img_numpy.shape)
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                                                spot_positions=pixel_candidates, output_path=out_path))
                pixel_coords_F.append(pixel_candidates)


    ## Next Steps: Project all pixel_coords_view back to 3D using pixel_to_mesh.
    ## Continued projections to be done only for I.C.E, F and T.H for now, since E.C/P.C are visible already in the view.
    ## Need to then check the filteration prompt for all the loading cases
    ## Need to train orthoviews to select orthographic views on even more cad data (ideally as much as possible!! Train on as much data as possible and add positional embeddings to thhe views after arranging them in that order)

    # --- 3D Raycasting (VECTORIZED) ---
    mark_points_ICE, mark_points_F, mark_points_TH =[], [], []
    for view, point_list_ICE, point_list_F, point_list_TH in tqdm(zip(gridded_views, pixel_coords_ICE, pixel_coords_F, pixel_coords_TH), total=len(gridded_views), desc="Mapping ICE to 3D Mesh", file=sys.stdout):

        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'

        out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8

        for edge_category, point_list in zip(['I.C.E', 'F', 'T.H'], [point_list_ICE, point_list_F, point_list_TH]):
            
            edge_list = point_list
            if len(edge_list) == 0: continue

            u_arr =[px[0] for px in edge_list]
            v_arr =[px[1] for px in edge_list]

            up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,-1)
            all_hits = pixel_to_mesh(cam_pos=cam_pos, F_pos=F_pos, u=u_arr, v=v_arr, 
                                    up_cam_vec=np.array(up_cam_vec), parallel_scale=parallel_scale,
                                    img_H=1000, img_W=1000, mesh=mesh, orthographic=True, return_all_hits=True)

            mark_points = []
            for p_world in all_hits:
                if p_world is not None and p_world.shape[0] >= 2:
                    for j in range(0, p_world.shape[0] - 1, 2):
                        if j>0: break
                        p_enter = p_world[j]
                        p_exit = p_world[j+1]
                        edge_points_on_mesh = p_enter + (p_exit - p_enter) * np.linspace(0, 1, round(np.linalg.norm(p_exit - p_enter)/point_density))[:, None]
                        mark_points.extend(edge_points_on_mesh)
                        
            if edge_category == 'I.C.E':
                mark_points_ICE.extend(mark_points)
            elif edge_category == 'F':
                mark_points_F.extend(mark_points)
            elif edge_category == 'T.H':
                mark_points_TH.extend(mark_points)
    
    mark_points_EC_PC =[]
    for view, point_list_EC_PC in tqdm(zip(gridded_views, pixel_coords_EC_PC), total=len(gridded_views), desc="Mapping EC/PC to 3D Mesh", file=sys.stdout):

        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'

        out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'
        elevation, azimuth = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(elevation, azimuth, mesh_path=mesh_file_path)
        parallel_scale = view_radius * 0.8
        
        if len(point_list_EC_PC) == 0: continue
        
        u_arr =[px[0] for px in point_list_EC_PC]
        v_arr =[px[1] for px in point_list_EC_PC]

        up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,-1)
        all_hits = pixel_to_mesh(cam_pos=cam_pos, F_pos=F_pos, u=u_arr, v=v_arr, 
                                up_cam_vec=np.array(up_cam_vec), parallel_scale=parallel_scale,
                                img_H=1000, img_W=1000, mesh=mesh, orthographic=True, return_all_hits=True)

        for p_world in all_hits:
            if p_world is not None and p_world.shape[0] >= 2:
                mark_points_EC_PC.append(p_world[0]) # Only take the first intersection point for EC/PC since they are visible in the view and we want to avoid noise from deeper ray intersections. For ICE, F and TH we will consider all ray intersections since they may not be visible in the view and we want to capture potential hidden features.


    ## After raycasting time to filter out cells
    # --- Render Output ---
    
    points_3d = filter_points_by_density_fast(refinement_points, point_density=point_density)  # Adjust threshold as needed
    render_mesh_views(mesh_file_path, output_dir=f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints_{experiment_name}', 
                    orthographic=False, points_3d=points_3d, verbose=True, add_axes=False, opacity=0.7)

    ## ---- Filter views based on visibility of detected features ----
    filteration_views = []
    ## Right now doing it only for fillets since they are open to select 
    points_3d_F = filter_points_by_density_fast(np.array(mark_points_F), point_density=point_density)
            
    for view in gridded_views:
        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view)}'
        
        el_angle, az_angle = extract_el_az_from_viewpath(view_path)
        ortho_angle_pairs = get_perpendicular_views(el_angle, az_angle)
        ## Select views with good visibility of the detected features (will be used for subsequent filtering prompts to the LLM)
        valid_views = {}
        for el_az in ortho_angle_pairs:
            el_angle_ortho, az_angle_ortho = el_az
            #if not (el_angle_ortho == 0 and az_angle_ortho == 0): continue
            cam_pos, F_pos, view_radius = return_cam_position(el_angle_ortho, az_angle_ortho, mesh_path=mesh_file_path)
            
            ## Check visibility
            visible, avg_distance_to_visible = check_points_visibility(mesh, cam_pos, points_3d_F, point_density)
            num_visible = np.sum(visible)
            if num_visible > 0.4 * len(points_3d_F):  # If more than 40% of the detected fillet points are visible in this orthogonal view, we consider it a good candidate for the next prompt
                if (el_angle_ortho, az_angle_ortho) not in valid_views.keys():
                    valid_views[(el_angle_ortho, az_angle_ortho)] = avg_distance_to_visible
                else:
                    # If the view is already in valid_views, we can update the average distance if this new one is better (lower)
                    if avg_distance_to_visible < valid_views[(el_angle_ortho, az_angle_ortho)]:
                        valid_views[(el_angle_ortho, az_angle_ortho)] = avg_distance_to_visible
        
    ## Filter valid views based on angle between views ad avg_dist (lower is better)
    valid_views_sorted = sorted(valid_views.items(), key=lambda x: x[1], reverse=True)  # Sort by average distance to visible points in descending order (lower distance is better)
    ## Non max suppression based on angle difference (keep views that are at least 15 degrees apart)
    filteration_views = [valid_views_sorted[0][0]]  # Start with the best view
    for view, avg_dist in valid_views_sorted[1:]:
        if angle_btw_views(view, filteration_views[-1]) >= 90:  # If the angle between this view and the last selected view is greater than or equal to 15 degrees
            filteration_views.append(view)
    
    filteration_view_imgs = []
    for el_az in filteration_views:
        el_angle, az_angle = el_az
        ## Synthesize load and marked point visualizations
        render_mesh_views_with_load(mesh_file_path, output_dir_prefix=f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints&load_{experiment_name}',
                                    n_azimuth=[az_angle], n_elevation=[el_angle], orthographic=False, 
                                    points_3d=points_3d, verbose=True, add_axes=False, loading_type=load_case)
        filteration_view_imgs.append(f'{parent_dir}/{cad_object}/renders_pyvista_with_meshpoints&load_{experiment_name}_{load_case}/view_e{el_angle}_a{az_angle}.png')
    

    # --- Prompt Construction ---
    prompt = get_prompt_2(num_perspective_views=len(perspective_views), num_gridded_views=len(filteration_view_imgs), load_case=load_case)  # You would define this function to return your actual prompt text

    messages =[]
    inference_imgs =[]
    gridded_views = []
    for view_path in filteration_view_imgs:
        gridded_path = overlay_grid(view_path, gridx=gridx, gridy=gridy, font_scale=1.0, 
                                    font_color=(0,0,0), grid_color=(0,0,0), font_thickness=1, 
                                    line_thickness=1, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    arrow_cell=None, output_dir=output_grid_dir)
        gridded_views.append(gridded_path)
    
    img_paths = perspective_views + gridded_views
    for img_path in img_paths:
        with open(img_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            inference_imgs.append(encoded_img)

    content =[{"type": "text", "text": prompt}]
    content.extend([{"type": "image_url", "image_url": f"data:image/png;base64,{img}"} for img in inference_imgs])
    messages.append({"role": "user", "content": content})

    # --- Protected API Call ---
    print(f"Prompt Stage 2 Region Filteration: Waiting for OpenRouter API Response ({llm_name})...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": llm_name, "temperature": 0.0, "messages": messages},
            timeout=60  # Prevent infinite hanging
        )
        response.raise_for_status() # Catches 429, 502, etc.
        response_data = response.json()
        
        if 'choices' not in response_data:
            print(f"API Error Response format: {response_data}")
            sys.exit(1)
            
        output = response_data['choices'][0]['message']['content']
        print("Response Received:", output)
        
    except Exception as e:
        print(f"API Request Failed: {e}")
        sys.exit(1) # Exit cleanly to inform parent script

    pred_cells_lists = parse_geom_feature_cells(output, prompt_type='geomax', infer_views=filteration_view_imgs, feature_categories=['Cells'])
    points_to_remove =[]

    for view_path, pred_cells in zip(filteration_view_imgs, pred_cells_lists):
        img = cv2.imread(view_path)
        if img is None:
            print(f"Failed to load image for final visualization: {view_path}")
            continue
        
        ## Remove the points located in the pred_cells in these views
        el,az = extract_el_az_from_viewpath(view_path)
        cam_pos, F_pos, view_radius = return_cam_position(el, az, mesh_path=mesh_file_path)
        visible_idxs, _ = check_points_visibility(mesh, cam_pos, np.array(points_3d), point_density)    
        visible_points = np.array(points_3d)[visible_idxs]
        visible_pts_pixel_coords = space_to_pixel(visible_points, cam_pos, F_pos, up_cam_vec=np.array([0,1,0]) if 'view_e90' not in view_path and 'view_e-90' not in view_path else np.array([0,0,-1])
                                                  , parallel_scale=parallel_scale, img_H=1000, img_W=1000, orthographic=True)
        for pt, pixel in zip(visible_points, visible_pts_pixel_coords):
            if is_pixel_in_pred_cells(pixel, pred_cells['Cells'], gridx, gridy, img.shape):
                points_to_remove.append(pt)

    ## Filter off points_to_remove from points_3d and re-render the view with the remaining points for final visualization
    points_to_keep = remove_points_by_proximity(points_3d, points_to_remove, point_density)
    
    render_mesh_views(mesh_file_path, output_dir=f'{parent_dir}/{cad_object}/final_visualization_{experiment_name}', 
                orthographic=False, points_3d=points_to_keep, verbose=True, add_axes=False, opacity=0.7)
    
    refinement_points = np.array(points_to_keep)
    np.save(f'{parent_dir}/{cad_object}/refinement_points_{experiment_name}.npy', refinement_points)

    if len(refinement_points) == 0:
        print(f'No points detected for refinement for {os.path.basename(mesh_file_path)} with experiment {experiment_name}')
    
    
    print('Lets See!!')    


            


            

