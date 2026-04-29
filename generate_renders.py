import os, re, numpy as np
import sys
import pyvista as pv
import h5py
from tqdm import tqdm
from PIL import Image
import shutil
freecad_base = "/data/1bali/miniforge3/envs/multi_view_3DQA"
# Append FreeCAD's Python library paths
sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)
# Now you can import FreeCAD normally
import FreeCAD
import Part
import Import
import Mesh
import MeshPart
from concurrent.futures import ProcessPoolExecutor, as_completed


def render_mesh_views(mesh_file, output_dir="renders_pyvista_mesh_initial", n_azimuth=12, n_elevation=[-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90], orthographic=False, points_3d=None, add_axes=True, verbose=False, axes_size='normal', opacity=1.0):
    """
    Render multiple views of a mesh (.obj, .stl, etc.) showing mesh elements (faces & edges).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = pv.read(mesh_file)
    if mesh.n_points == 0:
        raise ValueError(f"Mesh file {mesh_file} is empty or invalid")

    # --- Strong edge overlay ---
    # step_file_path = mesh_file.replace('.obj', '.step').replace('.stl', '.step')
    # cad_edges = extract_cad_edges(step_file_path)
    # cad_polylines = cad_edges_to_polydata(cad_edges)

    mesh_clean = (
        mesh
        .clean(tolerance=1e-6)
        .merge_points()
        .compute_normals(auto_orient_normals=True, split_vertices=False)
    )

    dihedral_angle = 30.0
    cad_polylines = [
        mesh_clean.extract_feature_edges(
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
            feature_angle=dihedral_angle
        )]
    
    # Compute center and scale for camera placement
    bounds = mesh.bounds
    center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
    if orthographic: 
        zoom = 1.2
        edge_mark_factor = 0.003
    else: 
        zoom = 4.0
        edge_mark_factor = 0.001
    radius = np.linalg.norm([
        bounds[1]-bounds[0],
        bounds[3]-bounds[2],
        bounds[5]-bounds[4]
    ])/2 * zoom

    # Start headless rendering (for servers)
    pv.start_xvfb()

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
        for azimuth in azimuths:
            
            # Convert spherical coordinates to Cartesian camera position
            # camera must be placed in the opposite direction of view pointing to the center
            # But pyvista's increasing z-coordinate points inward into screen, so we invert the sign cam = center + radius * view_dir instead of cam = center - radius * view_dir
            cam_x = center[0] + radius * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
            cam_y = center[1] + radius * np.sin(np.radians(elevation))
            cam_z = center[2] + radius * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

            plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
            plotter.set_background("white")

            view_dir = np.array([center[0]-cam_x, center[1]-cam_y, center[2]-cam_z])
            # 🔹 ORTHOGRAPHIC PROJECTION
            if orthographic:
                plotter.enable_parallel_projection()
            
            # --- Base shaded mesh ---
            plotter.add_mesh(
                mesh,
                color="#cccccc",
                smooth_shading=False,
                opacity=opacity,
                lighting=True,
                specular=0.1,
                ambient=0.3,
                diffuse=0.9,
            )

            try:
                for line in cad_polylines:
                    plotter.add_mesh(
                        line.tube(radius=edge_mark_factor * radius),
                        color="black",
                        lighting=False,
                        opacity=opacity,
                    )
            except: 
                print(f'Could not find feature or boundary edges for {mesh_file}')
                # edges = mesh_clean.extract_all_edges()
                # plotter.add_mesh(
                #     edges,
                #     color="black",
                #     line_width=2,
                #     render_lines_as_tubes=True
                # )



            if points_3d is not None:
                points_3d = pv.PolyData(points_3d)
                plotter.add_points(
                    points_3d,
                    color="red",
                    point_size=15,
                    render_points_as_spheres=True,
                )
            # # --- Wireframe overlay (for clear mesh edges) ---
            # plotter.add_mesh(
            #     mesh,
            #     style="wireframe",
            #     color="black",
            #     line_width=0.5,
            #     opacity=1.0
            # )
            
            # --- Axis legend (compatible + larger) ---
            if add_axes:
                if axes_size == 'large':
                    plotter.add_axes(
                        interactive=False,
                        line_width=8,           # ← Thicker lines (was 5)
                        labels_off=False,
                        x_color="red",
                        y_color="green", 
                        z_color="blue",
                        viewport=(0.0, 0.0, 0.4, 0.4),  # ← Larger viewport (32% → 40%)
                        cone_radius=0.8,        # ← Bigger arrow heads
                        shaft_length=0.7,       # ← Longer shafts
                        tip_length=0.3,         # ← Longer tips
                        label_size=(0.5, 0.2),  # ← Bigger labels (height, width)
                        ambient=0.8,            # ← Brighter
                    )
                else:
                    plotter.add_axes(
                    interactive=False,
                    line_width=5,
                    labels_off=False,
                    x_color="red",
                    y_color="green",
                    z_color="blue",
                    viewport=(0.0, 0.0, 0.32, 0.35),  # makes it occupy 25% of the viewport
                    )
                plotter.show_axes()  # optional 3D axes at bottom-right

            if abs(elevation) == 90:
                up_vector = (0, 0, -1)  # Use Z-axis as the up vector for top/bottom views
            else:
                up_vector = (0, 1, 0)   # Normal Y-axis up vector for all other views

            plotter.camera_position = [(cam_x, cam_y, cam_z), center, up_vector]

            # 🔹 SCALE CONTROL (critical for ortho)
            if orthographic:
                plotter.camera.parallel_scale = radius * 0.8
            filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}.png")
            plotter.show(screenshot=filename)
            plotter.close()
            if verbose:
                print(f"✅ Saved {filename}")
            
            if abs(elevation) == 90: break  # No need to continue azimuth loop for top/bottom views since they look the same from all azimuths

    if verbose:
        print(f"✅ Finished rendering all mesh views to: {output_dir}")

def create_hatch_texture(size=512, line_width=8, spacing=32):
    """Creates a high-contrast black and white diagonal hatch pattern."""
    # Create white background
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    
    # Draw thick diagonal lines
    for diag in range(-size, size * 2, spacing):
        for w in range(line_width):
            # Draw y = x + diag
            x = np.arange(size)
            y = x + diag + w
            mask = (y >= 0) & (y < size)
            img[y[mask], x[mask]] = [0, 0, 0] # Pure black lines
            
    return pv.Texture(img)

def read_pos_file(path):
    sp_re = re.compile(r"SP\(([eE0-9\.\-,]+)\)\{([eE0-9\.\-]+)\}")
    coords, values = [], []
    with open(path, "r") as f:
        for line in f:
            m = sp_re.search(line)
            if m:
                pts = [float(x) for x in m.group(1).split(",")]
                val = float(m.group(2))
                coords.append(pts)
                values.append(val)
    return np.array(coords), np.array(values)

def render_pos_views(pos_file, output_dir="renders_pyvista", n_azimuth=12, n_elevation=3, zoom=4.5, suffix=None):
    os.makedirs(output_dir, exist_ok=True)
    coords, vals = read_pos_file(pos_file)
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    #vals = np.power(vals, 0.4)

    # match vis_pos.py color map
    # cmap = [
    #     [0.0, [0, 0, 130]],
    #     [0.25, [0, 150, 255]],
    #     [0.5, [255, 255, 255]],
    #     [0.75, [255, 150, 0]],
    #     [1.0, [130, 0, 0]],
    # ]
    # cmap = [
    #     [0.0, [0, 0, 130]],
    #     [0.05, [255, 255, 255]],
    #     [0.3, [255, 255, 255]],
    #     [0.5, [255, 255, 255]],
    #     [0.75, [255, 255, 255]],
    #     [1.0, [255, 255, 255]]
    # ]

    cmap = [
        [0.0, [0, 0, 130]],   # White start (replaces blue)
        [0.03, [255, 255, 255]],  # White continues
        [0.3, [255, 252, 252]],   # Subtle pinkish-white transition
        [0.5, [255, 240, 240]],   # Light red
        [0.75, [255, 200, 200]],  # Medium light red
        [1.0, [255, 0, 0]]        # Full red
    ]
    # Convert to matplotlib colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", [(a, np.array(b)/255) for a,b in cmap])

    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).max() * zoom

    pv.start_xvfb()  # enables headless EGL/OSMesa rendering
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")

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
        for azimuth in azimuths:
            plotter.clear()
            # Convert spherical coordinates to Cartesian camera position
            ## z-axis in pyvista points is outward from screen (Right hand rule i.e not inward and hence the revolving angle alpha=np.pi/2-azimuth)
            ## elevation is the angle from the +y axis down toward the x-z plane
            ## azimuth is the angle taken from the positive x-axis toward the positive z-axis
            alpha = np.pi/2 - np.radians(azimuth) # angle from z axis towards +x axis in the azimuthal plane y=0
            beta = np.pi/2 - np.radians(elevation) # angle from x-z plane up towards y axis
            cam_x = center[0] + radius * np.sin(beta) * np.cos(alpha)
            cam_y = center[1] + radius * np.cos(beta)
            cam_z = center[2] + radius * np.sin(beta) * np.sin(alpha)
            
            # Create new plotter each time (avoids PyVista _actors issue)
            # plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
            # plotter.set_background("black")

            cloud = pv.PolyData(coords)
            cloud["vals"] = vals
            plotter.add_mesh(cloud, scalars="vals", cmap=cmap, point_size=5.0, render_points_as_spheres=True, show_scalar_bar=False)

            # If the camera is directly above or below the object, the View vector is 
            # parallel to the Y-axis. We must change the Up vector to avoid a singularity.
            if abs(elevation) == 90:
                up_vector = (0, 0, -1)  # Use Z-axis as the up vector for top/bottom views
            else:
                up_vector = (0, 1, 0)   # Normal Y-axis up vector for all other views

            plotter.camera_position = [(cam_x, cam_y, cam_z), center, up_vector]

            if suffix:
                filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_no_colorbar_{suffix}.png") 
            else:
                filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_no_colorbar.png") 
            plotter.render()  # 👈 force redraw before screenshot
            plotter.screenshot(filename)
            print("Saved", filename)
        print(f"✅ Finished saving {n_azimuth} azimuth views at elevation {elevation:.1f}°")
        
    plotter.close()


# Set global theme to handle empty meshes gracefully
pv.global_theme.allow_empty_mesh = True

def create_dashed_line(start, end, L, n_segments=8, radius=0.006):
    """
    Robustly creates a 3D dashed line. 
    Reduced n_segments for shorter lines to keep dash spacing looking correct.
    """
    start = np.array(start)
    end = np.array(end)
    full_vec = end - start
    
    segments = []
    for i in range(n_segments):
        # 60% dash, 40% gap
        s = start + (i / n_segments) * full_vec
        e = start + ((i + 0.6) / n_segments) * full_vec
        segments.append(pv.Line(s, e))
    
    if not segments:
        return None
        
    combined = pv.merge(segments)
    return combined.tube(radius=L * radius)

def render_mesh_views_with_load(mesh_file, output_dir_prefix="renders_mesh", n_azimuth=12, n_elevation=3, 
                      orthographic=False, points_3d=None, add_axes=True, verbose=True, 
                      axes_size='normal', loading_type=None,
                      arrow_density=0.05,
                      arrow_size=0.08,
                      support_density=0.04,
                      support_size=0.02,
                      support_thickness=0.008):
    
    output_dir = f"{output_dir_prefix}" #if loading_type else output_dir_prefix
    os.makedirs(output_dir, exist_ok=True)
    
    # if verbose:
    #     print(f"▶️ Processing Mesh: {os.path.abspath(mesh_file)}")
    #     print(f"  Mode: {loading_type if loading_type else 'Clean View'}")

    mesh = pv.read(mesh_file)
    if mesh.n_points == 0: 
        if verbose: print(f"  ⚠️ Skipping {mesh_file}: Mesh has no points.")
        return "[SKIP] Empty Mesh"

    mesh_clean = (
        mesh.clean(tolerance=1e-6).merge_points()
        .compute_normals(auto_orient_normals=True, point_normals=True, split_vertices=False)
    )

    bounds = mesh.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    cx_glob, cy_glob, cz_glob = center
    L = max(xmax-xmin, ymax-ymin, zmax-zmin) if max(xmax-xmin, ymax-ymin, zmax-zmin) > 0 else 1.0
    
    arr_len = L * arrow_size
    radius = np.linalg.norm([xmax-xmin, ymax-ymin, zmax-zmin])/2 * (1.2 if orthographic else 4.0)
    edge_mark_factor = 0.003 if orthographic else 0.001
    load_color = "#00008B" # Dark Blue

    # --- Pre-process 3D Points ---
    pd_points = None
    if points_3d is not None and len(points_3d) > 0:
        pd_points = pv.PolyData(np.array(points_3d))

    loading_items = [] 

    if loading_type:
        def get_resampled_points(y_low, y_high, spacing_val):
            surface = mesh_clean.clip_box([xmin-L, xmax+L, y_low, y_high, zmin-L, zmax+L], invert=False)
            if surface.n_points == 0: return np.array([])
            surf_poly = surface.extract_surface()
            gx, gz = np.meshgrid(np.arange(xmin, xmax+(L*spacing_val), L*spacing_val),
                                 np.arange(zmin, zmax+(L*spacing_val), L*spacing_val))
            grid_pts = np.c_[gx.ravel(), np.full(gx.size, (y_low + y_high)/2), gz.ravel()]
            poly_grid = pv.PolyData(grid_pts)
            dist = poly_grid.compute_implicit_distance(surf_poly)
            mask = np.abs(dist["implicit_distance"]) < (L * spacing_val * 0.5)
            return grid_pts[mask]

        # --- 1. TOP SURFACE LOADS ---
        top_pts = get_resampled_points(ymax - L*0.02, ymax + L*0.02, arrow_density)
        
        if len(top_pts) > 0:
            cx_top = top_pts[:, 0].mean()
            cz_top = top_pts[:, 2].mean()

            if loading_type == 'compression':
                cloud = pv.PolyData(top_pts + [0, arr_len, 0])
                cloud['v'] = np.tile([0, -1, 0], (len(top_pts), 1))
                loading_items.append({'mesh': cloud.glyph(orient='v', scale=False, factor=arr_len), 'color': load_color})

            elif loading_type == 'torsion':
                # create_dashed_line is assumed to be defined elsewhere
                axis = create_dashed_line([cx_top, ymax - L*0.1, cz_top], 
                                          [cx_top, ymax + L*0.2, cz_top], L)
                if axis: loading_items.append({'mesh': axis, 'color': 'red'})
                
                pts_float = top_pts + np.array([0, L * 0.05, 0]) 
                vx, vz = (top_pts[:, 2] - cz_top), -(top_pts[:, 0] - cx_top)
                vecs = np.c_[vx, np.zeros_like(vx), vz]
                v_norms = np.linalg.norm(vecs, axis=1)[:, None]
                vecs = np.divide(vecs, v_norms, out=np.zeros_like(vecs), where=v_norms!=0)
                
                cloud = pv.PolyData(pts_float); cloud['v'] = vecs
                loading_items.append({'mesh': cloud.glyph(orient='v', scale=False, factor=arr_len*0.8), 'color': load_color})

            elif loading_type == 'bending':
                axis = create_dashed_line([cx_top, ymax, cz_top - L*0.15], 
                                          [cx_top, ymax, cz_top + L*0.15], L)
                if axis: loading_items.append({'mesh': axis, 'color': 'red'})

                pts_adj, vecs = [], []
                for p in top_pts:
                    if p[0] >= cx_top:
                        pts_adj.append(p + [0, arr_len, 0]); vecs.append([0, -1, 0])
                    else:
                        pts_adj.append(p); vecs.append([0, 1, 0])
                cloud = pv.PolyData(np.array(pts_adj)); cloud['v'] = np.array(vecs)
                loading_items.append({'mesh': cloud.glyph(orient='v', scale=False, factor=arr_len), 'color': load_color})

            elif loading_type == 'shear':
                cloud = pv.PolyData(top_pts + [0, L*0.05, 0])
                cloud['v'] = np.tile([1, 0, 0], (len(top_pts), 1))
                loading_items.append({'mesh': cloud.glyph(orient='v', scale=False, factor=arr_len), 'color': load_color})

        # --- 2. BOTTOM FIXED SUPPORT ---
        bottom_pts = get_resampled_points(ymin - L*0.02, ymin + L*0.02, support_density)
        if len(bottom_pts) > 0:
            h_val = L * support_size
            starts = bottom_pts + np.array([0, -L*0.003, 0]) 
            ends = starts + np.array([-h_val, -h_val, 0])
            pts_arr = np.vstack((starts, ends))
            N = len(bottom_pts)
            conn = np.c_[np.full(N, 2), np.arange(N), np.arange(N) + N].ravel()
            support_poly = pv.PolyData(pts_arr, lines=conn)
            if support_poly.n_points > 0:
                loading_items.append({'mesh': support_poly.tube(radius=L * support_thickness), 'color': 'black'})

    # --- RENDER ENGINE ---
    pv.start_xvfb()
    elevations = n_elevation if isinstance(n_elevation, list) else [-90 + (180/(n_elevation+1))*e for e in range(1, n_elevation+1)]
    azimuths = n_azimuth if isinstance(n_azimuth, list) else [(360/n_azimuth)*a for a in range(n_azimuth)]

    #if verbose: print(f"  ...Rendering views to {os.path.abspath(output_dir)}")

    for elevation in elevations:
        for azimuth in azimuths:
            cam_x = cx_glob + radius * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
            cam_y = cy_glob + radius * np.sin(np.radians(elevation))
            cam_z = cz_glob + radius * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

            plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
            plotter.set_background("white")
            if orthographic: plotter.enable_parallel_projection()
            
            # Base mesh
            plotter.add_mesh(mesh_clean, color="#cccccc", smooth_shading=False, opacity=1.0, lighting=True)
            
            # Feature edges (tubes)
            feat_edges = mesh_clean.extract_feature_edges(feature_angle=30)
            if feat_edges.n_points > 0:
                plotter.add_mesh(feat_edges.tube(radius=edge_mark_factor * radius), color="black", lighting=False)

            # --- ADDING 3D POINTS (RED DOTS) ---
            if pd_points is not None:
                plotter.add_points(
                    pd_points,
                    color="red",
                    point_size=15,  # Matches render_mesh_views
                    render_points_as_spheres=True,
                    lighting=False
                )

            # Loading arrows/supports
            for item in loading_items:
                if item['mesh'].n_points > 0:
                    plotter.add_mesh(item['mesh'], color=item['color'], lighting=False, ambient=1.0)

            # Camera Setup
            up_vector = (0, 0, -1) if abs(elevation) == 90 else (0, 1, 0)
            plotter.camera_position = [(cam_x, cam_y, cam_z), [cx_glob, cy_glob, cz_glob], up_vector]
            
            if orthographic: 
                plotter.camera.parallel_scale = radius * 0.8
                
            filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}.png")
            plotter.show(screenshot=filename)
            plotter.close()

            if elevation in [-90, 90]: break  # No need to continue azimuth loop for top/bottom views
            
    if verbose: print(f"✅ COMPLETED task for {loading_type}")
    return f"[DONE] {loading_type}"

# if __name__ == '__main__':
#     #/data/1bali/Other_LLM_projects/multi_view_3DQA/2D_FE_Mesh_2.jpg
#     path_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset' #"/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries"
#     #CAD_mesh_path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-meshes"
#     processed_files_, timeout_files_ = [], []
#     for CAD_Folder in tqdm(os.listdir(path_dir), total=len(os.listdir(path_dir))):

#         #if CAD_Folder not in ['Electrical_Parts_Servos_SG-90_Servo-sg90']: continue
#         ## Debugging for a specific folder
#         # if CAD_Folder not in ['Mechanical_Parts_Profiles_EN_DIN1025-5_IPE-Profiles_IPE-Profile_360_DIN1025-5_S235JR']:
#         #     continue

#         mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
#         render_mesh_views(mesh_obj_file,
#                           output_dir=f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial',
#                           n_azimuth=12,
#                           n_elevation=9, orthographic=True, add_axes=False)#zoom=4.5 if not orthographic else 1)

#     # mesh_obj_file = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/00520041/00520041_589de10a30b7761010f2c170_step_030.obj'

#     # render_mesh_views(mesh_obj_file,
#     #                       output_dir=f'./renders_pyvista_mesh_initial',
#     #                       n_azimuth=12,
#     #                       n_elevation=9, orthographic=True, add_axes=False)

def process_single_folder(args):
    CAD_Folder, path_dir, load_case = args

    try:

        mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
        os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista', exist_ok=True)
        os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial', exist_ok=True)

        for file in os.listdir(f'{path_dir}/{CAD_Folder}'):
            if file.endswith('.obj'):
                mesh_obj_file = f'{path_dir}/{CAD_Folder}/{file}'
                shutil.copy(mesh_obj_file, f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj')
    
        output_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh'
        mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'

        if not os.path.exists(mesh_obj_file):
            return f"[SKIP] {CAD_Folder} (no mesh)"

        render_mesh_views_with_load(
            mesh_obj_file,
            output_dir_prefix=output_dir,
            n_azimuth=12,
            n_elevation=[-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90],
            orthographic=True,
            add_axes=False,
            loading_type=load_case,
            verbose=True
        )

        return f"[DONE] {CAD_Folder}"

    except Exception as e:
        return f"[ERROR] {CAD_Folder}: {str(e)}"

if __name__ == '__main__':

    path_dir = './test_meshes_7.04.2026'
    cad_folders = os.listdir(path_dir)
    load_cases = ['torsion', 'bending', 'compression']
    tasks = [(folder, path_dir, load) for folder in cad_folders for load in load_cases]

    # for task in enumerate(tasks):
    #     print(f"Processing {task[0]} ({task[1]}/{len(tasks)})") 
    #     result = process_single_folder(task)
    #     print(result)

    # 🔥 Tune this carefully
    max_workers = 50   # start safe (VTK + XVFB heavy)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_folder, task) for task in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            print(f.result())


# def process_single_folder(args):
#     CAD_Folder, path_dir = args

#     try:

#         #mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
#         os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista', exist_ok=True)
#         os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial', exist_ok=True)

#         flag=0
#         for file in os.listdir(f'{path_dir}/{CAD_Folder}'):
#             if file.endswith('.obj'):
#                 mesh_obj_file = f'{path_dir}/{CAD_Folder}/{file}'
#                 shutil.copy(mesh_obj_file, f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj')
#                 flag=1
#                 break
        
#         if flag==0:
#             mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
        
#         output_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial'

#         if not os.path.exists(mesh_obj_file):
#             return f"[SKIP] {CAD_Folder} (no mesh)"

#         render_mesh_views(
#             mesh_obj_file,
#             output_dir=output_dir,
#             n_azimuth=12,
#             n_elevation=[-90,90],
#             orthographic=True,
#             add_axes=False
#         )

#         return f"[DONE] {CAD_Folder}"

#     except Exception as e:
#         return f"[ERROR] {CAD_Folder}: {str(e)}"

# if __name__ == '__main__':

#     # path_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal'
#     # cad_folders = os.listdir(path_dir)
#     # tasks = [(folder, path_dir) for folder in cad_folders]

#     tasks = []
#     with open('/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/val.log', 'r') as log_file:
#         for line in log_file:
#             folder_path = line.strip()
#             tasks.append((os.path.basename(folder_path), os.path.dirname(folder_path)))
    
#     # 🔥 Tune this carefully
#     max_workers = 10   # start safe (VTK + XVFB heavy)

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_single_folder, task) for task in tasks]

#         for f in tqdm(as_completed(futures), total=len(futures)):
#             print(f.result())
