import os, re, numpy as np
import plotly.graph_objects as go
import kaleido  # ensure installed: pip install -U kaleido
import plotly.io as pio
import pyvista as pv
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import concurrent.futures


def combine_mesh_pointcloud_views(mesh_dir, pc_dir, output_dir):
    """
    Combine corresponding mesh and point cloud render images side by side.

    mesh_dir: path to mesh renders (e.g., renders_pyvista_mesh)
    pc_dir: path to point cloud renders (e.g., renders_pyvista)
    output_dir: directory where combined images will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    mesh_images = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".png")])
    pc_images = sorted([f for f in os.listdir(pc_dir) if f.endswith(".png")])

    common = sorted(list(set(mesh_images).intersection(set(pc_images))))
    if not common:
        print(f"No matching images found between {mesh_dir} and {pc_dir}")
        return

    for img_name in common:
        mesh_img = Image.open(os.path.join(mesh_dir, img_name)).convert("RGB")
        pc_img = Image.open(os.path.join(pc_dir, img_name)).convert("RGB")

        # Make sure both are same height
        h = max(mesh_img.height, pc_img.height)
        new_w = mesh_img.width + pc_img.width

        combined = Image.new("RGB", (new_w, h), (255, 255, 255))
        combined.paste(mesh_img, (0, 0))
        combined.paste(pc_img, (mesh_img.width, 0))

        out_path = os.path.join(output_dir, img_name)
        combined.save(out_path)
        print(f"✅ Saved combined view: {out_path}")

    print(f"✅ Finished combining {len(common)} image pairs into {output_dir}")

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

def render_mesh_views(mesh_file, output_dir="renders_mesh", n_azimuth=12, n_elevation=3, zoom=1.5):
    """
    Render multiple views of a mesh (.obj, .stl, etc.) showing mesh elements (faces & edges).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = pv.read(mesh_file)
    if mesh.n_points == 0:
        raise ValueError(f"Mesh file {mesh_file} is empty or invalid")

    # Compute center and scale for camera placement
    bounds = mesh.bounds
    center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
    radius = np.linalg.norm([
        bounds[1]-bounds[0],
        bounds[3]-bounds[2],
        bounds[5]-bounds[4]
    ]) * zoom

    # Start headless rendering (for servers)
    pv.start_xvfb()

    for e in range(n_elevation):
        elevation = -40 + (80 / max(n_elevation - 1, 1)) * e
        for a in range(n_azimuth):
            azimuth = (360 / n_azimuth) * a

            cam_x = center[0] - radius * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
            cam_y = center[1] - radius * np.sin(np.radians(elevation))
            cam_z = center[2] - radius * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

            plotter = pv.Plotter(off_screen=True, window_size=[500, 500])
            plotter.set_background("white")

            # --- Base shaded mesh ---
            plotter.add_mesh(
                mesh,
                color="#cccccc",          # light gray faces
                smooth_shading=True,
                opacity=1.0,
                lighting=True,
                specular=0.1,
                ambient=0.3,
                diffuse=0.9,
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

            # Camera setup
            plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 1, 0)]

            filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}.png")
            plotter.show(screenshot=filename)
            plotter.close()
            print(f"✅ Saved {filename}")

    print(f"✅ Finished rendering all mesh views to: {output_dir}")

def optimal_gamma(vals, gamma_range=(0.2, 8.0), n_steps=200, metric="variance"):
    """Find gamma that maximizes contrast (variance or entropy)."""
    vals = np.clip((vals - vals.min()) / (vals.max() - vals.min() + 1e-9), 0, 1)
    gammas = np.linspace(*gamma_range, n_steps)
    scores = []

    for g in gammas:
        v_t = vals**g
        if metric == "variance":
            score = np.var(v_t)
        elif metric == "entropy":
            hist, _ = np.histogram(v_t, bins=100, range=(0, 1), density=True)
            hist = hist[hist > 0]
            score = -np.sum(hist * np.log(hist))
        else:
            raise ValueError("metric must be 'variance' or 'entropy'")
        scores.append(score)

    best_idx = np.argmax(scores)
    return gammas[best_idx], scores[best_idx]


def optimal_gamma_top_tail(values, tail_frac=0.10, gamma_range=(0.2, 8.0), n_steps=300):
    """
    Finds gamma that maximizes Cohen's d between top `tail_frac` fraction and the rest.
    Returns best_gamma, best_score.
    """
    v = np.clip(values, a_min=0.0, a_max=None)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_steps)
    best_gamma, best_score = None, -np.inf
    n = v.size
    tail_count = max(1, int(np.ceil(tail_frac * n)))

    for g in gammas:
        vt = v ** g
        cutoff = np.partition(vt, -tail_count)[-tail_count]
        top = vt[vt >= cutoff]
        rest = vt[vt < cutoff]
        if top.size == 0 or rest.size == 0:
            continue

        m_top, m_rest = top.mean(), rest.mean()
        s_top, s_rest = top.std(ddof=0), rest.std(ddof=0)
        pooled_std = np.sqrt(((top.size - 1) * s_top**2 + (rest.size - 1) * s_rest**2) /
                             (top.size + rest.size - 2) + 1e-12)
        d = (m_top - m_rest) / (pooled_std + 1e-12)

        if d > best_score:
            best_score = d
            best_gamma = g

    return best_gamma, best_score

# def render_pyvista_views(
#     pos_file,
#     output_dir="renders_pyvista",
#     n_azimuth=12,
#     n_elevation=3,
#     zoom=1.5,
# ):
#     os.makedirs(output_dir, exist_ok=True)
#     coords, vals = read_pos_file(pos_file)
    
#     # --- Find optimal gamma for max contrast ---
#     gamma_opt, score = optimal_gamma(vals, metric='variance')
#     print(f"🎯 Optimal γ = {gamma_opt:.2f} (contrast score = {score:.4f})")

#     vals = np.clip((vals - vals.min()) / (vals.max() - vals.min() + 1e-9), 0, 1)
#     vals = vals**gamma_opt  # Apply optimal gamma

#     # Normalize after gamma correction
#     vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

#     # --- High contrast color map for white background ---
#     # Low = light gray/white (low stress)
#     # High = dark navy/black (high stress)
#     cmap = [
#         [0.0, [255, 255, 255]],   # white for low stress
#         [0.25, [230, 230, 230]],  # very light gray
#         [0.5, [50, 50, 50]],   # gray
#         [0.75, [20, 20, 20]],  # bluish gray
#         [1.0, [10, 10, 10]],         # black for high stress
#     ]

#     from matplotlib.colors import LinearSegmentedColormap
#     cmap = LinearSegmentedColormap.from_list(
#         "inverted_contrast", [(a, np.array(b) / 255) for a, b in cmap]
#     )

#     center = coords.mean(axis=0)
#     radius = np.linalg.norm(coords - center, axis=1).max() * zoom

#     pv.start_xvfb()
#     for e in range(n_elevation):
#         elevation = -40 + (80 / max(n_elevation - 1, 1)) * e
#         for a in range(n_azimuth):
#             azimuth = (360 / n_azimuth) * a

#             cam_x = center[0] + radius * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
#             cam_y = center[1] + radius * np.sin(np.radians(elevation))
#             cam_z = center[2] + radius * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

#             plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])
#             plotter.set_background("white")

#             cloud = pv.PolyData(coords)
#             cloud["vals"] = vals
#             plotter.add_mesh(
#                 cloud,
#                 scalars="vals",
#                 cmap=cmap,
#                 point_size=6.0,
#                 render_points_as_spheres=True,
#                 show_scalar_bar=False,
#             )

#             plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 1, 0)]

#             filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_inverted.png")
#             plotter.show(screenshot=filename)
#             plotter.close()
#             print(f"✅ Saved {filename}")

#         print(f"✅ Finished {n_azimuth} azimuths at elevation {elevation:.1f}°")


def render_pyvista_views(pos_file, output_dir="renders_pyvista", n_azimuth=12, n_elevation=3, zoom=1.5):
    os.makedirs(output_dir, exist_ok=True)
    coords, vals = read_pos_file(pos_file)
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    vals = np.power(vals, 0.4)

    # match vis_pos.py color map
    # cmap = [
    #     [0.0, [0, 0, 130]],
    #     [0.25, [0, 150, 255]],
    #     [0.5, [255, 255, 255]],
    #     [0.75, [255, 150, 0]],
    #     [1.0, [130, 0, 0]],
    # ]
    cmap = [
        [0.0, [0, 0, 130]],
        [0.25, [0, 150, 255]],
        [0.5, [255, 255, 255]],
        [0.75, [255, 255, 255]],
        [1.0, [255, 255, 255]]
    ]
    # Convert to matplotlib colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", [(a, np.array(b)/255) for a,b in cmap])

    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).max() * zoom

    pv.start_xvfb()  # enables headless EGL/OSMesa rendering
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")

    for e in range(n_elevation):
        elevation = -40 + (80 / max(n_elevation - 1, 1)) * e  # from -40° to +40°
        
        for a in range(n_azimuth):
            azimuth = (360 / n_azimuth) * a
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

            filename = os.path.join(output_dir, f"view_e{elevation:.0f}_a{azimuth:.0f}_no_colorbar.png")
            plotter.render()  # 👈 force redraw before screenshot
            plotter.screenshot(filename)
            print("Saved", filename)
        print(f"✅ Finished saving {n_azimuth} azimuth views at elevation {elevation:.1f}°")
        
    plotter.close()

        


# Example usage
if __name__ == "__main__":

    path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries"
    CAD_mesh_path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-meshes"
    
    for CAD_Folder in tqdm(os.listdir(path_dir), total=len(os.listdir(path_dir))):
        pos_file_path = f'{path_dir}/{CAD_Folder}/{CAD_Folder}_grad_0.pos'
        mesh_file_path = f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj'
        if not os.path.exists(pos_file_path):
            print(f"pos file not exist: {pos_file_path}")
            continue
        
        # if CAD_Folder not in ['Mechanical_Parts_Chains_Plate_Wheel_ISO_606_Simplex_½x⅛_Plate_Wheel_simplex_½x⅛']:
        #     continue
        pc_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista'
        mesh_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista_mesh'
        combined_dir = f'{path_dir}/{CAD_Folder}/renders_combined'
        pc_dir_new = f'{path_dir}/{CAD_Folder}/renders_pyvista_new'
        
        # render_pyvista_views(
        # pos_file_path,
        # output_dir=pc_dir_new,
        # n_azimuth=12,
        # n_elevation=5,
        # zoom=4.5)
        try:
            render_pyvista_views(
            pos_file_path,
            output_dir=pc_dir_new,
            n_azimuth=12,
            n_elevation=5,
            zoom=4.5)
        
            # render_mesh_views(
            # mesh_file_path,
            # output_dir=mesh_dir,
            # n_azimuth=12,
            # n_elevation=5,
            # zoom=2.0)
        
            # combine_mesh_pointcloud_views(mesh_dir, pc_dir, combined_dir)
            
            with open('valid_pos_files.txt', 'w') as f:
                f.write(f'{pos_file_path}\n')
        except:
            print(f'Could not process {pos_file_path}')
            with open('invalid_pos_files.txt', 'w') as f:
                f.write(f'{pos_file_path}\n')
