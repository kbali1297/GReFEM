api_key = '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'
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

def extract_az_el_from_viewpath(view_path):
    # Example view_path: '.../renders/view_e-30_a45.png'
    filename = os.path.basename(view_path)
    name_part = filename.split('.')[0].split('_')[0]  # 'view_e-30_a45'
    
    # Extract el and az using string manipulation
    el_str = name_part.split('e')[1].split('_')[0]  # '-30'
    az_str = name_part.split('a')[1]  # '45'
    
    return float(el_str), float(az_str)

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

def pixel_to_mesh_(cam_pos,
    F_pos,
    u, v,
    up_cam_vec,
    img_H, img_W,
    fov_in_degrees=75,
    parallel_scale=None,   # REQUIRED if orthographic=True
    mesh=None,
    orthographic=False,
    return_all_hits=False,
    return_no_location=False):
    """
    Convert pixel (u, v) to 3D world coordinate on mesh or view plane.
    Supports both perspective and orthographic projections.
    """
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
        plane_depth = np.linalg.norm(F_pos - cam_pos)

        fov_y_rad = math.radians(fov_in_degrees)
        fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
        fx = fy * img_W / img_H

        x_plane = (u - cx) / fx * plane_depth
        y_plane = -(v - cy) / fy * plane_depth

        ray_origin = cam_pos + x_plane * x_cam + y_plane * y_cam
        ray_dir = z_cam

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

        # Projection plane through focal point
        P_plane = F_pos + dx * x_cam + dy * y_cam

        ray_origin = P_plane + 1000000.0 * z_cam
        ray_dir = -z_cam

    # -------------------------------
    # Ray–mesh intersection
    # -------------------------------
    if mesh is None:
        return ray_origin + plane_depth * z_cam

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=np.array([ray_origin]),
        ray_directions=np.array([ray_dir])
    )


    if len(locations) == 0:
        if return_no_location:
            return None
        return np.array([ray_origin + plane_depth * z_cam])

    # sort hits by distance to camera
    dists = np.linalg.norm(locations - cam_pos, axis=1)
    order = np.argsort(dists)
    locations = locations[order]


    if return_all_hits:
        if len(locations.shape)>1:
            return locations
        return locations[None,:]
    else:
        return locations[0]   # closest hit


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


def return_depth_map(x0, y0, cell_width, cell_height, camera_position, object_center, view_radius, img_height, img_width, mesh):
    """
    Given a grayscale image patch, return a depth image where white pixels are closer (lower depth values)
    and black pixels are farther (higher depth values).
    """
    depth_map = np.full((cell_height, cell_width), np.inf)
    # Invert the grayscale values to represent depth
    for x in range(cell_width):
        for y in range(cell_height):
            u,v = x + x0, y + y0
            loc3D = pixel_to_mesh(cam_pos=camera_position, F_pos=object_center,
                          u=u, v=v, up_cam_vec=np.array([0,1,0]),parallel_scale=0.8*view_radius,
                          img_H=img_height, img_W=img_width, fov_in_degrees=30, mesh=mesh, orthographic=True, 
                          return_all_hits=False, return_no_location=True)
            if len(loc3D) > 0:
                depth_map[y,x] = loc3D[2] - camera_position[2]  # Z-coordinate as depth
            else:
                depth_map[y,x] = np.inf  # No intersection, set to infinity

    return depth_map

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

def edge_detection_contour_(image_path, cell_num, gridx=10, gridy=10, angle_thresh=30):
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

    # if cell_num==58:
    #     print('wait')
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
    #cnt = max(contours, key=cv2.contourArea)
    corners = []
    for contour in contours:
        if len(contour) < 20:
            continue
        # Check for real corner
        corners.extend(detect_corner_by_avg(contour, x0, y0, window=4, angle_thresh_deg=angle_thresh))

    inside_corners = []
    ## The corners must lie inside the CAD geometry (i.e. on mask)
    ## Object is white (1) on black (0) background
    if corners is not None:
        for corner in corners:
            x, y = corner
            flag=0
            shift_x, shift_y = 0, 0
            inspect_r = 6 
            num_theta = 72
            iter_points = [[x + inspect_r * np.cos(theta), y + inspect_r * np.sin(theta)] for theta in np.linspace(0,2*np.pi,num_theta)]
            num_points_out = len(iter_points)
            num_points_in = 0
            for iter_p in iter_points:
                iter_x, iter_y = int(round(iter_p[0])), int(round(iter_p[1]))
                if mask[iter_y, iter_x] == 1:
                    num_points_in += 1
                    shift_x += iter_x - x
                    shift_y += iter_y - y
                    num_points_out-= 1
            if num_points_in > round(len(iter_points) * (180+angle_thresh)/360) and num_points_out>0:
                shift_x/=num_points_in
                shift_y/=num_points_in
                corner += np.array([shift_x, shift_y]) # * 0.6                
                inside_corners.append(corner)
            # for dx in [-5, 5]:
            #     for dy in [-5, 5]:
            #         if mask[y + dy, x + dx] == 1:
            #             flag+=1
            #         else:
            #             # Shift the corner closer to geometry in case it does 
            #             # turn out to be internal corner
            #             shift_x, shift_y = -dx, -dy
            # if flag==3:
            #     corner += 0.6 * np.array([shift_x, shift_y])                
            #     inside_corners.append(corner)
    
    return inside_corners

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
    ## The corners must lie inside the CAD geometry (i.e. on mask)
    ## Object is white (1) on black (0) background
    # if corners is not None:
    #     for corner in corners:
    #         x, y = corner
    #         shift_x, shift_y = 0, 0
    #         inspect_r = 6 
    #         num_theta = 72
    #         iter_points = [[x + inspect_r * np.cos(theta), y + inspect_r * np.sin(theta)] for theta in np.linspace(0, 2*np.pi, num_theta)]
    #         num_points_out = len(iter_points)
    #         num_points_in = 0
            
    #         for iter_p in iter_points:
    #             iter_x, iter_y = int(round(iter_p[0])), int(round(iter_p[1]))
                
    #             # BOUNDARY CHECK ADDED HERE
    #             if 0 <= iter_x < w and 0 <= iter_y < h:
    #                 if mask[iter_y, iter_x] == 1:
    #                     num_points_in += 1
    #                     shift_x += iter_x - x
    #                     shift_y += iter_y - y
    #                     num_points_out -= 1

    #         if num_points_in >= round(len(iter_points) * (180 + angle_thresh) / 360): #and num_points_out > 0
    #             shift_x /= num_points_in
    #             shift_y /= num_points_in
    #             corner += np.array([shift_x, shift_y]) # * 0.6                
    #             inside_corners.append(corner)
    
    # return inside_corners

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

def circle_edge_support(edge_img, circle, tol=2.0, n_samples=360, min_support=0.3):
    """
    edge_img: binary edge image (Canny output)
    circle: (x, y, r)
    tol: radial tolerance in pixels
    min_support: fraction of points that must hit edges
    """

    h, w = edge_img.shape
    x0, y0, r = circle

    hits = 0
    for t in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
        x = int(round(x0 + r * np.cos(t)))
        y = int(round(y0 + r * np.sin(t)))

        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        # check small radial band
        for dr in range(-int(tol), int(tol) + 1):
            xx = int(round(x0 + (r + dr) * np.cos(t)))
            yy = int(round(y0 + (r + dr) * np.sin(t)))
            if 0 <= xx < w and 0 <= yy < h and edge_img[yy, xx]:
                hits += 1
                break

    return hits / n_samples >= min_support

def circle_in_img_patch(circle, image_path, cell_num, gridx=10, gridy=10):
    # _ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # h, w = _.shape
    h,w = 1000, 1000    
    cell_h = h // gridy
    cell_w = w // gridx

    r = (cell_num - 1) // gridx
    c = (cell_num - 1) % gridx

    x0, x1 = c * cell_w, (c + 1) * cell_w
    y0, y1 = r * cell_h, (r + 1) * cell_h

    circle_pts = [(circle['xc'] + circle['r'] * np.cos(theta), circle['yc'] + circle['r'] * np.sin(theta)) for theta in np.linspace(0,2*np.pi, num=12)]

    for pts in circle_pts:
        x, y = pts
        if x>=x0 and x<=x1 and y>=y0 and y<=y1:
            return True
    
    return False

def add_circle_pts_(circle, store_array):

    circle_pts = [(circle['xc'] + circle['r'] * np.cos(theta), circle['yc'] + circle['r'] * np.sin(theta)) for theta in np.linspace(0,2*np.pi, num=12)]
    circle_center = np.array([circle['xc'], circle['yc']])
    min_dist = 100000        
    if len(store_array['circles']) == 0:
        store_array['circles'].append(circle)
        store_array['pts'].extend(circle_pts)
    else:
        for idx, stored_circle in enumerate(store_array['circles']):
            stored_circle_center = np.array([stored_circle['xc'], stored_circle['yc']])    
            dist = np.linalg.norm(stored_circle_center - circle_center)
            if dist < min_dist:
                min_idx = idx
                min_dist = dist
        
        min_circle = store_array['circles'][min_idx]
        nearest_circle_center = np.array([min_circle['xc'], min_circle['yc']])
        
        if np.linalg.norm(nearest_circle_center - circle_center) > 10 \
            or np.abs(min_circle['r'] - circle['r']) > 10: 
            store_array['circles'].append(circle)
            store_array['pts'].extend(circle_pts)

            return store_array
    
    return store_array

def add_circle_pts(circle, store_array):
    THETAS = np.linspace(0, 2*np.pi, 12, endpoint=False)
    COS_T = np.cos(THETAS)
    SIN_T = np.sin(THETAS)
    xc, yc, r = circle['xc'], circle['yc'], circle['r']

    # vectorized circle points
    circle_pts = np.column_stack((
        xc + r * COS_T,
        yc + r * SIN_T
    ))

    try:
        if not store_array['circles']:
            store_array['circles'].append(circle)
            store_array['pts'].extend(circle_pts)
            return store_array

        # extract centers & radii in one pass
        centers = np.array([[c['xc'], c['yc']] for c in store_array['circles']])
        radii   = np.array([c['r'] for c in store_array['circles']])

        # vectorized distances
        deltas = centers - np.array([xc, yc])
        dists = np.sqrt(np.sum(deltas * deltas, axis=1))

        min_idx = np.argmin(dists)

        if dists[min_idx] > 10 or abs(radii[min_idx] - r) > 10:
            store_array['circles'].append(circle)
            store_array['pts'].extend(circle_pts)

        
    except:
        store_array.extend(circle_pts)

    return store_array



import cv2
import numpy as np

def detect_all_circles_cv2(image_path, min_arc_ratio=0.20): # Ensure this isn't overridden to 0.1 in your args!
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    blurred_img = cv2.GaussianBlur(img, (7,7), 1.5)
    # blurred_img = img
    edges = cv2.Canny(blurred_img, 80, 160)
    
    if not np.any(edges):
        edges = cv2.Canny(blurred_img, 10, 40)
        param1_val = 40
    else:
        param1_val = 120

    # 1. FINELY TUNED MULTI-SCALE BRACKETS
    # By grouping radii closely, we ensure `param2` scales properly with the circumference.
    # We demand roughly 20-25% of the circle's perimeter to exist in edge pixels.
    max_dim = min(h, w)
    brackets =[
        # Tiny holes (R: 4-15) -> Circ: ~25-94 pixels. Need ~12 edge pixels.
        {'minR': 4,   'maxR': 15,  'param2': 10,  'minDist': 2},
        
        # Small fillets (R: 15-30) -> Circ: ~94-188 pixels. Need ~25 edge pixels.
        {'minR': 15,  'maxR': 30,  'param2': 25,  'minDist': 2},
        
        # Medium curves (R: 30-60) -> Circ: ~188-376 pixels. Need ~45 edge pixels.
        {'minR': 30,  'maxR': 60,  'param2': 45,  'minDist': 2},
        
        # Large arches (R: 60-120) -> Circ: ~376-750 pixels. Need ~80 edge pixels.
        {'minR': 60,  'maxR': 120, 'param2': 60,  'minDist': 2},

        # Massive boundary curves (R: 120+) -> Need ~120+ edge pixels.
        {'minR': 120, 'maxR': int(round(0.6 * max_dim)), 'param2': 100, 'minDist': 2}
    ]

    raw_circles =[]

    # Run HoughCircles for each size bracket
    for b in brackets:
        if b['minR'] >= b['maxR']: continue 
        
        c = cv2.HoughCircles(
            blurred_img, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=b['minDist'],
            param1=param1_val, param2=b['param2'],
            minRadius=b['minR'], maxRadius=b['maxR']
        )
        if c is not None:
            raw_circles.extend(c[0])

    valid_circles =[]

    # Sort circles by radius (smallest first). 
    raw_circles = sorted(raw_circles, key=lambda x: x[2])
    
    for x, y, r in raw_circles:
        dynamic_tol = max(1.5, min(3.0, r * 0.05)) 
        
        if circle_edge_support(edges, (x, y, r),
                               tol=dynamic_tol,
                               n_samples=360,
                               min_support=min_arc_ratio):
            
            is_duplicate = False
            for vx, vy, vr in valid_circles:
                dist = np.hypot(x - vx, y - vy)
                
                # 2. FIXED DE-DUPLICATION LOGIC (NMS)
                # If the centers are very close (within 10 pixels or 20% of radius)...
                if dist < max(10.0, vr * 0.20):
                    # ... AND their radii are very similar (within 15% difference), it's a duplicate.
                    # BUT if radii are significantly different (concentric), this fails and both survive!
                    if abs(r - vr) < max(5.0, vr * 0.15):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                valid_circles.append((x, y, r))

    # Format output
    circles =[]
    #valid_circles = raw_circles
    for vc in valid_circles:
        circles.append({'xc': vc[0], 'yc': vc[1], 'r': vc[2]})

    return circles

def detect_all_circles_cv2_1(image_path, min_arc_ratio=0.25):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    #blurred_img = cv2.GaussianBlur(img, (7,7), 1.5)
    blurred_img=  img
    edges = cv2.Canny(blurred_img, 80, 160)
    
    if not np.any(edges):
        edges = cv2.Canny(blurred_img, 10, 40)
        param1_val = 40
    else:
        param1_val = 120

    # 1. MULTI-SCALE SEARCH BRACKETS
    # Dictionary of: {minRadius, maxRadius, param2 (votes), minDist}
    brackets =[
        # Small holes: need very few votes, can be close together
        {'minR': 4,  'maxR': 20,  'param2': 14, 'minDist': 10},
        
        # Medium features (fillets, standard holes): need more votes to reject noise
        {'minR': 20, 'maxR': 80,  'param2': 25, 'minDist': 20},
        
        # Large curves (arches, outer boundaries): need many votes
        {'minR': 80, 'maxR': int(round(0.6*min(h,w))), 'param2': 45, 'minDist': 40}
    ]
    
    raw_circles = []
    
    # Run HoughCircles for each size bracket
    for b in brackets:
        if b['minR'] >= b['maxR']: continue 
        
        c = cv2.HoughCircles(
            blurred_img, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=b['minDist'],
            param1=param1_val, param2=b['param2'],
            minRadius=b['minR'], maxRadius=b['maxR']
        )
        if c is not None:
            raw_circles.extend(c[0])

    valid_circles =[]

    # 2. VALIDATION & DE-DUPLICATION (NMS)
    for x, y, r in raw_circles:
        
        # Dynamic Tolerance: 2 pixels is a huge error for a radius=4 hole (50% error), 
        # but very strict for a radius=100 curve. Scale it!
        dynamic_tol = max(1.5, min(3.0, r * 0.05)) 
        
        if circle_edge_support(edges, (x, y, r),
                               tol=dynamic_tol,
                               n_samples=360,
                               min_support=min_arc_ratio):
            
            # Check if this circle is a duplicate/overlapping heavily with an already saved one
            is_duplicate = False
            for vx, vy, vr in valid_circles:
                dist = np.hypot(x - vx, y - vy)
                # If centers are very close AND radii are similar, consider it noise/duplicate
                if dist < (vr * 0.4) and abs(r - vr) < (vr * 0.3):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_circles.append((x, y, r))

    # Format output
    circles =[]
    for vc in valid_circles:
        circles.append({'xc': vc[0], 'yc': vc[1], 'r': vc[2]})
    
    return circles

def detect_all_circles_cv2_1(image_path, min_arc_ratio=0.2): # 1. Lowered to 0.2 to allow 25% (quarter) circles
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Blur once to use for both Canny and Hough
    blurred_img = cv2.GaussianBlur(img, (7,7), 1.5)
    
    # Edges are ONLY used for your custom circle_edge_support validation
    edges = cv2.Canny(blurred_img, 80, 160)
    
    if not np.any(edges):
        edges = cv2.Canny(blurred_img, 10, 40)
        
        circles_cv2 = cv2.HoughCircles(
            blurred_img, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=10, # 2. minDist changed from 0.1 to 10.0 pixels
            param1=40, param2=15,
            minRadius=4, maxRadius=int(round(0.6*min(h,w)))
        )
    else:
        circles_cv2 = cv2.HoughCircles(
            blurred_img, cv2.HOUGH_GRADIENT, # 3. CRITICAL: Passed blurred_img, NOT edges!
            dp=1.0, minDist=10,            # minDist changed to 10.0 pixels
            param1=120,                      # Slightly lowered to catch fainter holes
            param2=20,                       # 4. CRITICAL: Lowered from 65 to 20 to catch small/partial circles
            minRadius=4,                     # Dropped to 4 to catch tiny holes
            maxRadius=int(round(0.6*min(h,w)))
        )
    
    valid_circles =[]

    if circles_cv2 is not None:
        circles_cv2 = circles_cv2[0]  # unwrap
        for x, y, r in circles_cv2:
            if circle_edge_support(edges, (x, y, r),
                                tol=2.0,
                                n_samples=360,
                                min_support=min_arc_ratio): # Now allows >= 20% of a circle
                valid_circles.append((x, y, r))

    if not valid_circles: 
        circles = []
    else:
        circles =[]
        for valid_circle in valid_circles:
            circles.append({'xc': valid_circle[0],
                            'yc': valid_circle[1],
                            'r' : valid_circle[2]})
    
    return circles


def detect_all_circles_cv2_(image_path, min_arc_ratio=0.2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    edges = cv2.Canny(
          cv2.GaussianBlur(img, (7,7), 1.5),
          80, 160)
    
    if np.any(edges)==False:
        edges = cv2.Canny(
            cv2.GaussianBlur(img, (7,7), 1.5),
            10, 40)
        
        circles_cv2 = cv2.HoughCircles(
            img, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=0.1,
            param1=40, param2=15,
            minRadius=5, maxRadius=int(round(0.6*min(h,w)))
        )
    else:
        circles_cv2 = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=0.1,
            param1=160, param2=65,
            minRadius=5, maxRadius=int(round(0.6*min(h,w)))
        )
    
    valid_circles = []

    if circles_cv2 is not None:
        circles_cv2 = circles_cv2[0]  # unwrap
        for x, y, r in circles_cv2:
            if circle_edge_support(edges, (x, y, r),
                                tol=2.0,
                                n_samples=360,
                                min_support=min_arc_ratio): # Allow partial circles, arc angle = min_arc_ratio * 360
                valid_circles.append((x, y, r))

    if valid_circles is None: circles = []
    else:
        circles = []
        for valid_circle in valid_circles:
            circles.append({'xc': valid_circle[0],
                            'yc': valid_circle[1],
                            'r' : valid_circle[2]})
    
    return circles

def return_cam_position(view_img, mesh_path, zoom=1.2):
    
    mesh = pv.read(mesh_path)
    bounds = mesh.bounds
    object_center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
    view_radius = np.linalg.norm([
        bounds[1]-bounds[0],
        bounds[3]-bounds[2],
        bounds[5]-bounds[4]
    ])/2 * zoom

    ## Azimuth and elevation from the image filename
    azimuth =   int(view_img.split('/')[-1].split('.')[0].split('_a')[1].split('_')[0])
    elevation = int(view_img.split('/')[-1].split('.')[0].split('_e')[1].split('_')[0])
    
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
    for pt in candidate['p_on']:
        if (patch_x_min <= pt[0] < patch_x_max and patch_y_min <= pt[1] < patch_y_max):
            return True
    return False

def return_img_patch(patch_num, grid_res):
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

def parse_geom_feature_cells(output, prompt_type, infer_views, grid_res, cons_cell_lookup=0, feature_categories=['I.C.E', 'B.H', 'T.H']):
    pred_cells = {category: [] for category in feature_categories}
    if prompt_type in['geomax', 'geomid']:
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

def extract_all_contours_geometry(img, min_contour_len=30, sampling_step=15, pixel_offset=5):
    """
    Extracts (p_on, p_in, p_out) triplets for ALL valid contours in the entire image.
    The number of samples is dynamically scaled based on the contour's length.
    
    Args:
        img: The full input image (BGR or Grayscale).
        min_contour_len: Minimum perimeter length to be considered a valid feature.
        sampling_step: A sample is taken roughly every 'sampling_step' pixels along the contour.
        pixel_offset: Distance in pixels for p_in and p_out from p_on.
        
    Returns:
        List of dictionaries, where each dictionary contains geometry data for one contour.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    
    # Use RETR_LIST to get all contours (both outer boundaries and inner holes)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    all_geometries = []
    
    for c in contours:
        if len(c) < min_contour_len:
            continue
            
        # 1. Dynamic Sampling: Scale number of samples with contour length
        num_samples = max(5, len(c) // sampling_step)
        
        # 2. Find Centroid: Used to consistently point 'p_in' towards the interior
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroid = np.array([cx, cy])
        else:
            # Fallback for degenerate contours
            centroid = np.mean(c[:, 0, :], axis=0)
            
        # 3. Determine if straight (A line has ~0 area compared to its length)
        area = cv2.contourArea(c)
        length = cv2.arcLength(c, closed=True)
        is_straight = area < (length * 1.5) # Threshold for "straightness/flatness"
        
        points_on, points_in, points_out = [], [], []
        
        # Sample points evenly across the contour
        indices = np.linspace(0, len(c) - 1, num_samples, dtype=int)
        
        for idx in indices:
            p_on = c[idx][0]
            
            # Calculate local tangent and normal (handle wrap-around for closed contours)
            idx_prev = (idx - 3) % len(c)
            idx_next = (idx + 3) % len(c)
            pt1, pt2 = c[idx_prev][0], c[idx_next][0]
            
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            tangent_len = math.hypot(dx, dy)
            if tangent_len == 0: continue
            
            nx, ny = -dy / tangent_len, dx / tangent_len
            normal = np.array([nx, ny])
            
            # Align local normal to point towards the centroid (defining the "inside")
            vec_to_centroid = centroid - p_on
            if np.dot(normal, vec_to_centroid) > 0:
                dir_in = normal
            else:
                dir_in = -normal
                
            p_in = p_on + dir_in * pixel_offset
            p_out = p_on - dir_in * pixel_offset
            
            points_on.append(p_on)
            points_in.append(p_in)
            points_out.append(p_out)

        if not points_on:
            continue

        all_geometries.append({
            "p_on": np.array(points_on),   
            "p_in": np.array(points_in),   
            "p_out": np.array(points_out), 
            "is_straight": is_straight,
            "contour_ref": c  # Keep a reference to the original contour for plotting
        })

    return all_geometries

def detect_extrusion_boundary_or_protruding_boundary_vectorized(
    geom, mesh, cam_pos, F_pos, up_cam_vec=(0,1,0), img_H=1000, img_W=1000, orthographic=True,
    fov_in_degrees=75, parallel_scale=None,
    tol=1e-3, inf_threshold=1e5, consensus_ratio=0.4
):
    all_pixels = np.vstack([geom['p_in'], geom['p_on'], geom['p_out']])
    all_u, all_v = all_pixels[:, 0], all_pixels[:, 1]
    num_samples = len(geom['p_on'])

    all_hits_3d = pixel_to_mesh(
        cam_pos=cam_pos, F_pos=F_pos, u=all_u, v=all_v,
        up_cam_vec=up_cam_vec, img_H=img_H, img_W=img_W,
        fov_in_degrees=fov_in_degrees, parallel_scale=parallel_scale,
        mesh=mesh, orthographic=orthographic, return_no_location=True
    )
    
    # Safely compute depths, assigning infinity if ray missed
    all_depths = np.array([
        np.linalg.norm(hit - cam_pos) if hit is not None else np.inf 
        for hit in all_hits_3d
    ])

    d_ins = all_depths[:num_samples]
    d_ons = all_depths[num_samples : 2 * num_samples]
    d_outs = all_depths[2 * num_samples:]

    success_mask = np.zeros(num_samples, dtype=bool)
    valid_mask = (d_ins < inf_threshold) & (d_ons < inf_threshold) & (d_outs < inf_threshold)
    
    if np.any(valid_mask):
        v_in, v_on, v_out = d_ins[valid_mask], d_ons[valid_mask], d_outs[valid_mask]

        #cond_step_down = (np.abs(v_in - v_on) <= tol) & (v_out > v_in + tol)   
        cond_step_in   = (np.abs(v_out - v_on) <= tol) & (v_in > v_out + tol)  
        cond_step_out =  (np.abs(v_out - v_on) <= tol) & (v_in < v_out - tol)
        cond_valley    = (v_on > v_in + tol) & (v_on > v_out + tol)            
        #cond_ridge     = (v_on < v_in - tol) & (v_on < v_out - tol)            

        success_mask[valid_mask] = cond_step_out | cond_step_in | cond_valley #| cond_ridge
    
    ratio = np.sum(success_mask) / num_samples
    return ratio >= consensus_ratio # else np.array([]) geom['p_on'][success_mask] if 


def detect_through_hole(
    geom, mesh, cam_pos, F_pos, up_cam_vec=(0,1,0), img_H=1000, img_W=1000, orthographic=True,
    fov_in_degrees=75, parallel_scale=None,
    inf_threshold=1e5, consensus_ratio=0.4
):
    if geom['is_straight']:
        return np.array([]) 

    all_pixels = np.vstack([geom['p_in'], geom['p_out']])
    all_u, all_v = all_pixels[:, 0], all_pixels[:, 1]
    num_samples = len(geom['p_in'])

    all_hits_3d = pixel_to_mesh(
        cam_pos=cam_pos, F_pos=F_pos, u=all_u, v=all_v,
        up_cam_vec=up_cam_vec, img_H=img_H, img_W=img_W,
        fov_in_degrees=fov_in_degrees, parallel_scale=parallel_scale,
        mesh=mesh, orthographic=orthographic, return_no_location=True
    )
    
    all_depths = np.array([
        np.linalg.norm(hit - cam_pos) if hit is not None else np.inf 
        for hit in all_hits_3d
    ])

    d_ins = all_depths[:num_samples]
    d_outs = all_depths[num_samples:]

    # Success: inside hits infinity, outside hits mesh
    success_mask = (d_ins > inf_threshold) & (d_outs < inf_threshold)
    
    ratio = np.sum(success_mask) / num_samples
    return  ratio >= consensus_ratio # else np.array([]) geom['p_on'][success_mask] if

def detect_parts_of_contour(contours, cell_list, gridx, gridy, img_shape):
    contour_parts = []
    pixel_coords = []
    for candidate in contours:
        for cell_num in cell_list:
            for pt in candidate['p_on']:
                if is_contour_in_patch({'p_on': [pt]}, cell_num, gridx, gridy, img_shape):
                    pixel_coords.append(pt)
    return pixel_coords
    


# --- MOCK DATA FOR DEMONSTRATION ---
if __name__ == '__main__':
    # You would replace this with your actual mesh and camera setup
    parent_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/test_meshes_7.04.2026'
    cad_object = '00210097'
    load_case = 'torsion'
    num_views = 2
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
    print(f"Waiting for OpenRouter API Response ({llm_name})...")
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

    pred_cells_lists = parse_geom_feature_cells(output, prompt_type='geomax', infer_views=infer_views_paths, grid_res=(gridx, gridy), cons_cell_lookup=1, feature_categories=['I.C.E', 'E.C/P.C', 'T.H', 'F'])
    
    output_2d_dir = f'{parent_dir}/{cad_object}/renders_pyvista_with_2Dpoints_{experiment_name}'
    os.makedirs(output_2d_dir, exist_ok=True)
    
    pixel_coords_ICE, pixel_coords_EC_PC, pixel_coords_TH, pixel_coords_F = [], [], [], []
    output_img_paths = set()  # Use a set to avoid duplicates
    for i, view_path in enumerate(gridded_views):
        view_path = f'{parent_dir}/{cad_object}/renders_pyvista_mesh_initial/{os.path.basename(view_path)}'
        
        out_path = f'{output_2d_dir}/{os.path.basename(view_path)}'

        cam_pos, F_pos, view_radius = return_cam_position(view_path, mesh_path=mesh_file_path)
        up_cam_vec = (0,1,0) if 'view_e-90' not in view_path and 'view_e90' not in view_path else (0,0,1)
        img = cv2.imread(view_path)
        if img is None:
            print(f"Failed to load image: {view_path}")
            continue
        
        img_numpy = np.array(img)
        geom = extract_all_contours_geometry(img_numpy)
        ec_pc_candidates, th_candidates =[], []
        for contour in geom:
            if detect_extrusion_boundary_or_protruding_boundary_vectorized(geom=contour, mesh=mesh, cam_pos=cam_pos, F_pos=F_pos, parallel_scale=0.8 * view_radius, up_cam_vec=up_cam_vec):  # You would fill in the actual mesh and camera parameters here
                ec_pc_candidates.append(contour)
            elif detect_through_hole(geom=contour, mesh=mesh, cam_pos=cam_pos, F_pos=F_pos, parallel_scale=0.8 * view_radius, up_cam_vec=up_cam_vec):  # You would fill in the actual mesh and camera parameters here
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
    
                for patch_num in patch_nums:
                    ## convert image to numpy array and extract the patch corresponding to patch_num
                    patch = return_img_patch(patch_num, grid_res=(gridx, gridy))
                    candidates = edge_detection_contour(view_path, patch_num, gridx, gridy, angle_thresh=20)
                    pixels_coords_view.extend(candidates)
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 255, 0), 
                                                spot_positions=pixels_coords_view, output_path=out_path))
                pixel_coords_ICE.append(pixels_coords_view)
    
            elif category == 'E.C/P.C':
                print(f"Analyzing Extruded Contour / Portruding Contour in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for extrusion boundaries / protruding boundaries
                    
                for patch_num in patch_nums:
                    patch = return_img_patch(patch_num, grid_res=(gridx, gridy))
                    for candidate in ec_pc_candidates:
                        
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixels_coords_view.extend(candidate['p_on'])
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                                                spot_positions=pixels_coords_view, output_path=out_path))
                pixel_coords_EC_PC.append(pixels_coords_view)

            elif category == 'T.H':
                print(f"Analyzing Through Holes / Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for through holes / fillets
            
                for patch_num in patch_nums:
                    patch = return_img_patch(patch_num, grid_res=(gridx, gridy))
                    for candidate in th_candidates:
                        if is_contour_in_patch(candidate, patch_num, gridx, gridy, img_numpy.shape):
                            pixels_coords_view.extend(candidate['p_on'])
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(0, 0, 255), 
                                                spot_positions=pixels_coords_view, output_path=out_path))
                pixel_coords_TH.append(pixels_coords_view)
            
            elif category == 'F':
                print(f"Analyzing Fillets in {view_path} for patches {patch_nums}")
                # Similar patch extraction logic for fillets
            
                pixels_coords_view.extend(detect_parts_of_contour(geom, patch_nums, gridx, gridy, img_numpy.shape))
                
                output_img_paths.add(mark_spots_in_image(view_path, spot_radius=4, spot_color=(255, 0, 0), 
                                                spot_positions=pixels_coords_view, output_path=out_path))
                pixel_coords_F.append(pixels_coords_view)


            ## Next Steps: Project all pixel_coords_view back to 3D using pixel_to_mesh, then analyze the 3D points to classify the type of feature (e.g., is it really a concave edge, an extrusion boundary, etc.) based on their spatial arrangement and relation to the mesh geometry.
            ## Continued projections to be done only for F and T.H for now, since I.C.E and E.C/P.C are more edge-like and may not have as clear 3D point clusters.
            ## Need to then check the filteration prompt for all the loading cases
            ## Need to train orthoviews to select orthographic views on even more cad data (ideally as much as possible!!)
    

