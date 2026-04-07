import os, re, numpy as np
import sys
import pyvista as pv
import h5py
from tqdm import tqdm
from PIL import Image
import shutil
freecad_base = "/data/1bali/GReFEM_env"
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


def render_mesh_views(mesh_file, output_dir="renders_mesh", n_azimuth=12, n_elevation=3, orthographic=False, points_3d=None, add_axes=True, verbose=False, axes_size='normal'):
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
                smooth_shading=True,
                opacity=0.7,
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
                        opacity=1.0,
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

            world_up = np.array([0.0, 1.0, 0.0])

            # # Project world_up onto the image plane
            # up = world_up - np.dot(world_up, view_dir) * view_dir
            # if np.linalg.norm(up) < 1e-6:
            #     # Degenerate case: looking straight up/down
            #     up = np.array([0.0, 1.0, 0.0])
            # else:
            #     up /= np.linalg.norm(up)

            # Camera setup
            plotter.camera_position = [(cam_x, cam_y, cam_z), center, world_up]

            # 🔹 SCALE CONTROL (critical for ortho)
            if orthographic:
                plotter.camera.parallel_scale = radius * 0.8
            filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}.png")
            plotter.show(screenshot=filename)
            plotter.close()
            if verbose:
                print(f"✅ Saved {filename}")

    if verbose:
        print(f"✅ Finished rendering all mesh views to: {output_dir}")

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

            plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 1, 0)]

            if suffix:
                filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_no_colorbar_{suffix}.png") 
            else:
                filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_no_colorbar.png") 
            plotter.render()  # 👈 force redraw before screenshot
            plotter.screenshot(filename)
            print("Saved", filename)
        print(f"✅ Finished saving {n_azimuth} azimuth views at elevation {elevation:.1f}°")
        
    plotter.close()

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
    CAD_Folder, path_dir = args

    try:

        #mesh_obj_file = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
        os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista', exist_ok=True)
        os.makedirs(f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial', exist_ok=True)

        for file in os.listdir(f'{path_dir}/{CAD_Folder}'):
            if file.endswith('.obj'):
                mesh_obj_file = f'{path_dir}/{CAD_Folder}/{file}'
                shutil.copy(mesh_obj_file, f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj')
        
        output_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh_initial'

        if not os.path.exists(mesh_obj_file):
            return f"[SKIP] {CAD_Folder} (no mesh)"

        render_mesh_views(
            mesh_obj_file,
            output_dir=output_dir,
            n_azimuth=12,
            n_elevation=9,
            orthographic=True,
            add_axes=False
        )

        return f"[DONE] {CAD_Folder}"

    except Exception as e:
        return f"[ERROR] {CAD_Folder}: {str(e)}"

if __name__ == '__main__':

    path_dir = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/test_meshes_rebuttal'
    cad_folders = os.listdir(path_dir)

    tasks = [(folder, path_dir) for folder in cad_folders]

    # 🔥 Tune this carefully
    max_workers = 50   # start safe (VTK + XVFB heavy)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_folder, task) for task in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            print(f.result())
