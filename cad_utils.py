import os, numpy as np
import sys
import pyvista as pv
import cv2
import trimesh
import math
freecad_base = "/data/1bali/miniforge3/envs/vtk_offscreen"
# Append FreeCAD's Python library paths
sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)
# Now you can import FreeCAD normally
import FreeCAD
import Part
import Import
import Mesh
import MeshPart


def tessellate_entity_with_offset(
    shape,
    entity,
    entity_type,
    offset_dist,
    tessellation=0.05,
    samples=80,
):
    """
    Tessellate and offset a CAD entity (face or edge) along outward normals.

    Args:
        shape: FreeCAD shape
        entity: Part.Face or Part.Edge
        entity_type: 'face' or 'edge'
        offset_dist: absolute offset distance
        tessellation: face tessellation
        samples: edge discretization samples

    Returns:
        pv.PolyData or None
    """

    bbox = shape.BoundBox
    model_center = np.array([bbox.Center.x, bbox.Center.y, bbox.Center.z])

    # ==========================
    # FACE
    # ==========================
    if entity_type == "face":
        face = entity
        surface = face.Surface

        mesh = MeshPart.meshFromShape(
            Shape=face,
            LinearDeflection=tessellation,
            AngularDeflection=0.523,
            Relative=False,
        )

        raw_verts = [FreeCAD.Vector(v.x, v.y, v.z) for v in mesh.Topology[0]]
        displaced_verts = []

        for v in raw_verts:
            try:
                u, v_param = surface.parameter(v)
                n = surface.normal(u, v_param)

                if face.Orientation == "Reversed":
                    n.multiply(-1.0)

                n.normalize()
                dv = v.add(n.multiply(offset_dist))
                displaced_verts.append([dv.x, dv.y, dv.z])

            except Exception:
                displaced_verts.append([v.x, v.y, v.z])

        faces = []
        for tri in mesh.Topology[1]:
            if len(tri) == 3:
                faces.append([3, tri[0], tri[1], tri[2]])

        if len(faces) == 0:
            return None

        return pv.PolyData(
            np.array(displaced_verts),
            np.array(faces).flatten(),
        )

    # ==========================
    # EDGE
    # ==========================
    elif entity_type == "edge":
        edge = entity

        u0, u1 = edge.ParameterRange
        params = np.linspace(u0, u1, samples)

        pts = []
        for u in params:
            try:
                p = edge.valueAt(u)
                pts.append(np.array([p.x, p.y, p.z]))
            except Exception:
                continue

        if len(pts) < 2:
            return None

        pts = np.array(pts)

        faces = shape.ancestorsOfType(edge, Part.Face)
        displaced_pts = []

        for p in pts:
            best_normal = None
            best_dot = -np.inf

            for face in faces[:2]:
                try:
                    surface = face.Surface
                    u_face, v_face = surface.parameter(
                        FreeCAD.Vector(p[0], p[1], p[2])
                    )

                    n = surface.normal(u_face, v_face)

                    if face.Orientation == "Reversed":
                        n.multiply(-1.0)

                    n_vec = np.array([n.x, n.y, n.z])
                    n_vec /= np.linalg.norm(n_vec)

                    dot = np.dot(n_vec, p - model_center)
                    if dot > best_dot:
                        best_dot = dot
                        best_normal = n_vec

                except Exception:
                    continue

            if best_normal is None:
                fallback = p - model_center
                best_normal = fallback / np.linalg.norm(fallback)

            displaced_pts.append(p + offset_dist * best_normal)

        displaced_pts = np.array(displaced_pts)

        poly = pv.PolyData(displaced_pts)
        poly.lines = np.hstack(
            [[len(displaced_pts)], np.arange(len(displaced_pts))]
        )
        return poly

    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

def farthest_point_sampling(points, k, start_idx=None):
    """
    Farthest Point Sampling (FPS)

    Args:
        points: (N, D) numpy array of points
        k: number of points to sample
        start_idx: optional int, starting point index

    Returns:
        sampled_points: (k, D) numpy array
        sampled_indices: (k,) numpy array of indices
    """
    N, D = points.shape
    if k > N: k=N

    sampled_indices = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf)

    if start_idx is None:
        start_idx = np.random.randint(N)

    sampled_indices[0] = start_idx
    current_point = points[start_idx]

    for i in range(1, k):
        dist = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices[i] = np.argmax(distances)
        current_point = points[sampled_indices[i]]

    return points[sampled_indices], sampled_indices

def highlight_topk_edges_offset(
    shape,
    k=5,
    samples=80,
    offset_ratio=0.001,
):
    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    edges_sorted = sorted(
        enumerate(shape.Edges),
        key=lambda x: x[1].Length,
        reverse=True
    )
    top_edges = edges_sorted[:k]

    edge_polys, edge_idxs = [], []

    for edge_idx, edge in top_edges:
        poly = tessellate_entity_with_offset(
                shape,
                edge,
                entity_type="edge",
                offset_dist=offset_dist,
                samples=samples,
            )
        if poly is not None:
            edge_polys.append(poly)
            edge_idxs.append(edge_idx)

    return edge_polys, edge_idxs


def highlight_cad_edge(
    shape,
    edge_idxs,
    samples=80,
    offset_ratio=0.001,
):
    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    edge_polys = []

    for edge_idx in edge_idxs:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Edges[edge_idx],
            entity_type="edge",
            offset_dist=offset_dist,
            samples=samples,
        )
        if poly is not None:
            edge_polys.append(poly)

    return edge_polys


def highlight_topk_cad_faces_offset(
    shape,
    k=5,
    tessellation=0.05,
    offset_ratio=0.003,
):
    """
    Highlight top-k largest CAD faces.
    """

    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    face_areas = [(i, f.Area) for i, f in enumerate(shape.Faces)]
    face_areas.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in face_areas[:k]]

    face_polys = []

    for fid in top_ids:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Faces[fid],
            "face",
            offset_dist,
            tessellation,
        )
        if poly is not None:
            face_polys.append(poly)

    return face_polys, top_ids

def highlight_cad_face(
    shape,
    face_idxs,
    tessellation=0.05,
    offset_ratio=0.003,
):
    """
    Highlight specific CAD faces.
    """

    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    face_polys = []

    for fid in face_idxs:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Faces[fid],
            "face",
            offset_dist,
            tessellation,
        )
        if poly is not None:
            face_polys.append(poly)

    return face_polys

def extract_cad_edges(shape, step_file=None):
    if step_file is not None: # Override the part input and read part from the step file
        shape = Part.read(step_file)

    cad_edges = []

    for edge in shape.Edges:
        faces = shape.ancestorsOfType(edge, Part.Face)

        # true CAD edge = exactly two faces meet
        if len(faces) == 2:
            cad_edges.append(edge)

    return cad_edges

def discretize_edge(edge, n=50):
    params = np.linspace(edge.FirstParameter, edge.LastParameter, n)
    return np.array([edge.valueAt(u) for u in params], dtype=float)

def cad_edges_to_polydata(edges, samples=50):
    polylines = []

    for edge in edges:
        pts = discretize_edge(edge, samples)

        line = pv.PolyData(pts)
        cells = np.hstack([[len(pts)], np.arange(len(pts))])
        line.lines = cells

        polylines.append(line)

    return polylines

def freecad_shape_to_pyvista(shape, linear_deflection=0.1, angular_deflection=0.523):
    """
    Convert a FreeCAD Part.Shape to PyVista PolyData (faces only).
    """
    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=linear_deflection,
        AngularDeflection=angular_deflection,
        Relative=False
    )

    # mesh.Topology[0] -> list[Base.Vector]
    verts = np.array([[v.x, v.y, v.z] for v in mesh.Topology[0]])

    faces = []
    for f in mesh.Topology[1]:
        if len(f) == 3:
            faces.append([3, f[0], f[1], f[2]])

    faces = np.array(faces).flatten()
    return pv.PolyData(verts, faces)


def freecad_edges_to_pyvista(shape, edge_samples=80):
    """
    Robust extraction of CAD edges using FreeCAD's discretize().
    Works for all curve types (lines, arcs, splines, trimmed curves).
    """
    polylines = []

    for edge in shape.Edges:
        try:
            pts_fc = edge.discretize(Num=edge_samples)
        except Exception:
            # Fallback: skip pathological edges
            continue

        if len(pts_fc) < 2:
            continue

        pts = np.array([[p.x, p.y, p.z] for p in pts_fc])

        lines = np.hstack([[len(pts)], np.arange(len(pts))])
        poly = pv.PolyData(pts)
        poly.lines = lines
        polylines.append(poly)

    return polylines


def render_cad_views(
    cad_file,
    output_dir=None,
    n_azimuth=12,
    n_elevation=3,
    orthographic=False,
    points_3d=None,
    add_axes=True,
    verbose=True,
    axes_size="normal",
    tessellation=0.1,
    highlight_edge_idxs=None,
    highlight_face_idxs=None,
):
    """
    Fast multi-view CAD renderer using FreeCAD + PyVista.

    Optimizations:
    - Single Plotter instance
    - Static geometry added once
    - Camera-only updates per view
    - screenshot() instead of show()
    """

    if output_dir is None:
        output_dir = os.path.dirname(cad_file)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load CAD
    # --------------------------------------------------
    doc = FreeCAD.newDocument()
    Import.insert(cad_file, doc.Name)
    doc.recompute()

    shapes = [
        obj.Shape
        for obj in doc.Objects
        if hasattr(obj, "Shape") and not obj.Shape.isNull()
    ]

    if len(shapes) == 0:
        raise ValueError(f"No valid shapes found in {cad_file}")

    shape = shapes[0]

    # --------------------------------------------------
    # Convert to PyVista
    # --------------------------------------------------
    cad_mesh = freecad_shape_to_pyvista(
        shape, linear_deflection=tessellation
    )

    cad_edges = extract_cad_edges(shape)
    cad_polylines = cad_edges_to_polydata(cad_edges)

    if highlight_edge_idxs is not None:
        chosen_edges = highlight_cad_edge(
            shape, edge_idxs=highlight_edge_idxs
        )

    if highlight_face_idxs is not None:
        chosen_faces = highlight_cad_face(
            shape, face_idxs=highlight_face_idxs
        )

    # --------------------------------------------------
    # Camera setup
    # --------------------------------------------------
    bounds = cad_mesh.bounds
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
    )

    diag = np.linalg.norm(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ]
    )

    zoom = 1.2 if orthographic else 4.0
    edge_mark_factor = 0.003 if orthographic else 0.001
    radius = diag / 2 * zoom

    elevations = (
        n_elevation
        if isinstance(n_elevation, list)
        else [
            -90 + (180 / (n_elevation + 1)) * e
            for e in range(1, n_elevation + 1)
        ]
    )

    azimuths = (
        n_azimuth
        if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )

    # --------------------------------------------------
    # Plotter (ONE instance)
    # --------------------------------------------------
    pv.start_xvfb()

    plotter = pv.Plotter(
        off_screen=True, window_size=(1024, 1024)
    )
    plotter.set_background("white")

    if orthographic:
        plotter.enable_parallel_projection()

    # --------------------------------------------------
    # Static geometry (added ONCE)
    # --------------------------------------------------
    plotter.add_mesh(
        cad_mesh,
        color="#cccccc",
        opacity=1.0,
        lighting=True,
        specular=0.2,
        ambient=0.3,
        diffuse=0.9,
    )

    # Precompute edge tubes once
    edge_tubes = [
        line.tube(
            radius=edge_mark_factor * radius,
            n_sides=6,
        )
        for line in cad_polylines
    ]

    for tube in edge_tubes:
        plotter.add_mesh(
            tube, color="black", lighting=False
        )

    if highlight_edge_idxs is not None:
        for edge in chosen_edges:
            plotter.add_mesh(
                edge.tube(
                    radius=edge_mark_factor * 3 * radius,
                    n_sides=8,
                ),
                color="red",
                lighting=False,
            )

    if highlight_face_idxs is not None:
        for face in chosen_faces:
            plotter.add_mesh(
                face,
                color="blue",
                opacity=1.0,
                smooth_shading=True,
            )

    if points_3d is not None:
        plotter.add_points(
            pv.PolyData(points_3d),
            color="red",
            point_size=15,
            render_points_as_spheres=True,
        )

    if add_axes:
        plotter.add_axes(
            interactive=False,
            line_width=5,
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(0.0, 0.0, 0.32, 0.35),
        )

    # --------------------------------------------------
    # Camera sweep + screenshots
    # --------------------------------------------------
    world_up = (0.0, 0.0, 1.0)

    for elevation in elevations:
        for azimuth in azimuths:

            cam_x = center[0] - radius * np.cos(
                np.radians(-elevation)
            ) * np.sin(np.radians(azimuth))
            cam_y = center[1] - radius * np.cos(
                np.radians(-elevation)
            ) * np.cos(np.radians(azimuth))
            cam_z = center[2] - radius * np.sin(
                np.radians(-elevation)
            )

            plotter.camera_position = [
                (cam_x, cam_y, cam_z),
                center,
                world_up,
            ]

            if orthographic:
                plotter.camera.parallel_scale = radius * 0.8

            if highlight_edge_idxs is not None and highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}&face{highlight_face_idxs}.png"
                )
            elif highlight_edge_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}.png"
                )
            elif highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_face{highlight_face_idxs}.png"
                )
            else:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}.png"
                )

            filename = os.path.join(output_dir, image_name)
            plotter.render()
            plotter.screenshot(filename)

            if verbose:
                print(f"✅ Saved {filename}")

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    plotter.close()
    FreeCAD.closeDocument(doc.Name)

    if verbose:
        print(
            f"✅ Finished rendering CAD views → {output_dir}"
        )

def convert_to_obj(input_file, output_file, consider_first_part_idx=None):
    ext = os.path.splitext(input_file)[1].lower()

    doc = FreeCAD.newDocument("Doc")

    try:
        if ext in [".step", ".stp", ".iges", ".igs"]:
            Part.open(input_file)  # directly loads B-rep shapes
        elif ext in [".fcstd"]:
            FreeCAD.open(input_file)
        else:
            Mesh.open(input_file)  # STL, OBJ, etc.
    except Exception as e:
        print(f"❌ Failed to open {input_file}: {e}")
        return

    meshes = []
    for obj_idx, obj in enumerate(FreeCAD.ActiveDocument.Objects):
        if consider_first_part_idx is not None:
            if consider_first_part_idx==obj_idx:
                try:
                    if obj.TypeId == "Mesh::Feature":
                        meshes.append(obj.Mesh)
                    elif hasattr(obj, "Shape") and not obj.Shape.isNull():
                        mesh = MeshPart.meshFromShape(
                            Shape=obj.Shape,
                            LinearDeflection=0.1,
                            AngularDeflection=0.5,
                            Relative=False
                        )
                        meshes.append(mesh)
                except Exception as e:
                    print(f"⚠️ Skipping {obj.Label}: {e}")
                break
        else: #plot all objects
            try:
                if obj.TypeId == "Mesh::Feature":
                    meshes.append(obj.Mesh)
                elif hasattr(obj, "Shape") and not obj.Shape.isNull():
                    mesh = MeshPart.meshFromShape(
                        Shape=obj.Shape,
                        LinearDeflection=0.1,
                        AngularDeflection=0.5,
                        Relative=False
                    )
                    meshes.append(mesh)
            except Exception as e:
                print(f"⚠️ Skipping {obj.Label}: {e}")
        
    if meshes:
        merged = Mesh.Mesh()  # ✅ new container
        for m in meshes:
            merged.addMesh(m.copy())  # ✅ avoid immutability
        merged.write(output_file)
        print(f"✅ Exported {input_file} → {output_file}")
    else:
        print(f"⚠️ No meshable objects found in {input_file}")

    return output_file

def compute_optimal_views_combined(cad_file,
                          target_edge_idxs=None,
                          target_face_idxs=None,
                          n_azimuth=12,
                          n_elevation=9,
                          orthographic=True):
    
    # Ensure lists are initialized
    target_edge_idxs = target_edge_idxs or []
    target_face_idxs = target_face_idxs or []

    doc = FreeCAD.newDocument()
    Import.insert(cad_file, doc.Name)
    doc.recompute()

    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and not obj.Shape.isNull():
            shapes.append(obj.Shape)

    if len(shapes) == 0:
        raise ValueError(f"No valid shapes found in {cad_file}")

    shape = shapes[0]

    # Convert CAD to mesh for raytracing
    mesh_file = convert_to_obj(cad_file, output_file=cad_file.replace(".step", ".obj"), consider_first_part_idx=0)
    mesh = trimesh.load(mesh_file)

    # ---------------------------
    # Extract points for ALL target edges and faces combined
    # ---------------------------
    all_target_points = []

    # Process Edges by specific indices
    for idx in target_edge_idxs:
        if idx >= len(shape.Edges):
            print(f"Warning: Edge index {idx} out of bounds.")
            continue
        cad_edge = shape.Edges[idx]
        # Discretize edge to get points (FreeCAD native way to get points)
        edge_pts = np.array([[v.x, v.y, v.z] for v in cad_edge.discretize(Number=50)])
        if len(edge_pts) > 0:
            cad_edge_pts_fps, _ = farthest_point_sampling(edge_pts, k=30)
            all_target_points.extend(cad_edge_pts_fps)

    # Process Faces by specific indices
    for idx in target_face_idxs:
        if idx >= len(shape.Faces):
            print(f"Warning: Face index {idx} out of bounds.")
            continue
        cad_face = shape.Faces[idx]
        # Tessellate face to get vertices and faces
        vertices, mesh_faces = cad_face.tessellate()
        vertices = np.array([[v.x, v.y, v.z] for v in vertices])
        mesh_faces = np.array(mesh_faces)
        
        if len(mesh_faces) > 0:
            face_centroids = vertices[mesh_faces].mean(axis=1)
            face_centroids_fps, _ = farthest_point_sampling(points=face_centroids, k=30)
            all_target_points.extend(face_centroids_fps)

    if not all_target_points:
        raise ValueError("No points could be extracted from the provided edge/face indices.")

    # ---------------------------
    # View Setup
    # ---------------------------
    elevations = (
        n_elevation if isinstance(n_elevation, list)
        else [-90 + (180 / (n_elevation + 1)) * e for e in range(1, n_elevation + 1)]
    )

    azimuths = (
        n_azimuth if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )
    
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    zoom = 1.2 if orthographic else 4.0
    radius = np.linalg.norm(bounds[0,:] - bounds[1,:]) / 2 * zoom

    # ---------------------------
    # View Inspection Loop (Evaluating Combined Visibility)
    # ---------------------------
    view_occlusion_scores = {}

    for elevation in elevations:
        for azimuth in azimuths:
            cam_x = center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
            cam_y = center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
            cam_z = center[2] - radius * np.sin(np.radians(-elevation))

            cam_pos = np.array([cam_x, cam_y, cam_z])

            if orthographic:
                z_cam = center - cam_pos
                base_ray_dir = z_cam / np.linalg.norm(z_cam)
            
            key = f'view_e{int(elevation)}_a{int(azimuth)}'
            view_occlusion_scores[key] = 0

            # Raycast against ALL target points at once
            for pt in all_target_points:
                if not orthographic:
                    ray_dir = (pt - cam_pos) / np.linalg.norm(pt - cam_pos)
                else:
                    ray_dir = base_ray_dir
                
                ray_origin = np.array(pt) - ray_dir * 10000.0
                
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=np.array([ray_origin]),
                    ray_directions=np.array([ray_dir])
                )

                try:
                    # If the intersection is closer to the ray origin than the target point, it's occluded
                    if np.linalg.norm(locations[0] - ray_origin) < np.linalg.norm(pt - ray_origin) - 1e-4:
                        view_occlusion_scores[key] += 1
                except: 
                    # If ray fails/errors, penalize
                    view_occlusion_scores[key] += 2

    # ---------------------------
    # Sorting and Formatting Output
    # ---------------------------
    # Sort views ascending by occlusion score (Lower occlusion = Higher visibility)
    sorted_views = sorted(view_occlusion_scores.items(), key=lambda item: item[1])
    
    cad_dirpath = os.path.dirname(cad_file)
    
    # Generate the prioritized list of views
    optimal_viewpaths = [
        f'{cad_dirpath}/{view_key}_combined_marked.png' 
        for view_key, score in sorted_views
    ]
    
    return optimal_viewpaths, target_edge_idxs, target_face_idxs

def compute_optimal_views(cad_file,
    n_azimuth=12,
    n_elevation=9,
    num_top_edges=1,
    num_top_faces=1,
    orthographic=True):

    doc = FreeCAD.newDocument()
    Import.insert(cad_file, doc.Name)
    doc.recompute()

    shapes = []

    for obj in doc.Objects:
        if hasattr(obj, "Shape") and not obj.Shape.isNull():
            shapes.append(obj.Shape)

    if len(shapes) == 0:
        raise ValueError(f"No valid shapes found in {cad_file}")


    # Create a compound (non-boolean, preserves topology)
    shape = shapes[0] #Part.makeCompound(shapes)

    cad_edges, cad_edge_idxs = highlight_topk_edges_offset(shape=shape, k=num_top_edges)
    cad_faces, cad_face_idxs = highlight_topk_cad_faces_offset(shape=shape, k=num_top_faces)

    
    mesh_file = convert_to_obj(cad_file, output_file=cad_file.replace(".step", ".obj"), consider_first_part_idx=0)

    #mesh = pv.read(mesh_file)
    mesh = trimesh.load(mesh_file)

    elevations = (
        n_elevation if isinstance(n_elevation, list)
        else [-90 + (180 / (n_elevation + 1)) * e for e in range(1, n_elevation + 1)]
    )

    azimuths = (
        n_azimuth if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )
    
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    if orthographic: 
        zoom = 1.2
    else: 
        zoom = 4.0
    

    radius = np.linalg.norm(bounds[0,:] - bounds[1,:])/2 * zoom
    pts_edge_list, face_centroids_fps_list = [], []
    for cad_edge in cad_edges:
        cad_edge_pts_fps, _ = farthest_point_sampling(cad_edge.points, k=30)
        # pts.extend(cad_edge_pts_fps)
        ## For multiple edges seperate lists of edge_pts
        pts_edge_list.append(cad_edge_pts_fps)

    for cad_face in cad_faces:
        mesh_faces = cad_face.faces.reshape(-1, 4)[:, 1:]
        pts_faces = cad_face.points

        face_centroids = pts_faces[mesh_faces].mean(axis=1)
        face_centroids_fps, _ = farthest_point_sampling(points=face_centroids, k=30)
        face_centroids_fps_list.append(face_centroids_fps)

    # ---------------------------
    # Views inspection loop for each cad edge and cad face, which views maximize visibility 
    # ---------------------------
    
    edge_viewpaths_list, face_viewpaths_list = [],[]
    for pts, face_centroids_fps in zip(pts_edge_list, face_centroids_fps_list):
        edge_occlude, face_occlude = {}, {}
        for elevation in elevations:
            for azimuth in azimuths:

                cam_x = center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
                cam_y = center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
                cam_z = center[2] - radius * np.sin(np.radians(-elevation))

                cam_pos = np.array([cam_x, cam_y, cam_z])

                ## Check if occluded, currently mechanism works for orthographic views
                # Create a ray to the points
                if orthographic:
                    z_cam = center - cam_pos
                    ray_dir = z_cam / np.linalg.norm(z_cam)
                #ray_dir = z_cam
                
                key = f'view_e{int(elevation)}_a{int(azimuth)}'
                edge_occlude[key] = 0
                face_occlude[key] = 0

                for pt in pts:
                    if not orthographic:
                        ray_dir = (pt - cam_pos) / np.linalg.norm(pt - cam_pos)
                    ray_origin = np.array(pt) - ray_dir * 10000.0
                    locations, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_dir])
                    )

                    try:
                        if np.linalg.norm(locations[0] - ray_origin) < np.linalg.norm(pt - ray_origin):
                            edge_occlude[key] += 1
                    except: edge_occlude[key] += 2

                for centroid in face_centroids_fps:
                    if not orthographic:
                        ray_dir = (centroid - cam_pos) / np.linalg.norm(centroid - cam_pos)
                    ray_origin = centroid - ray_dir * 10000.0
                    locations, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_dir])
                    )

                    try:
                        if np.linalg.norm(locations[0] - ray_origin) < np.linalg.norm(centroid - ray_origin):
                            face_occlude[key] += 1
                    except: face_occlude[key] += 2


        ## Sorting views according to occluded edge and face scores
        edge_occlude_ = [(key, value) for key, value in edge_occlude.items()]
        face_occlude_ = [(key, value) for key, value in face_occlude.items()]

        sorted_views_edge = sorted(edge_occlude_, key= lambda x: x[1])
        sorted_views_face = sorted(face_occlude_, key= lambda x: x[1])
        ## Choosing the top 3rd, 4th view
        cad_dirpath = os.path.dirname(cad_file)
        edge_viewpaths = [f'{cad_dirpath}/{edge_view[0]}_marked_edge.png' for edge_view in sorted_views_edge] 
        face_viewpaths = [f'{cad_dirpath}/{face_view[0]}_marked_face.png' for face_view in sorted_views_face]

        edge_viewpaths_list.append(edge_viewpaths)
        face_viewpaths_list.append(face_viewpaths)
    
    
    return edge_viewpaths_list, face_viewpaths_list, cad_edge_idxs, cad_face_idxs


def pixel_to_mesh(cam_pos,
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

def mark_spots_in_image(img_path, spot_radius=10, spot_color=(0, 0, 255), spot_positions=[], output_path=None):
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


if __name__ == '__main__':
    cad_file_path = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset/abc_0000_step_v00/00000001/00000001_1ffb81a71e5b402e966b9341_step_000.step'
    render_save_path = '/'.join(cad_file_path.split('/')[:-1])
    render_cad_views(cad_file=cad_file_path, output_dir=render_save_path, orthographic=True)


# def highlight_topk_cad_faces_offset(
#     shape,
#     k=5,
#     tessellation=0.05,
#     offset_ratio=0.003,
# ):
#     """
#     Highlight top-k largest CAD faces and push them outward along
#     their local normals so they are always visible (prevent z-fighting).
#     """

#     # --- Compute model scale for relative offset ---
#     bbox = shape.BoundBox
#     diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
#     offset_dist = diag * offset_ratio

#     # --- Sort faces by area ---
#     face_areas = [(i, f.Area) for i, f in enumerate(shape.Faces)]
#     face_areas.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [i for i, _ in face_areas[:k]]

#     all_verts = []
#     all_faces = []
#     vert_offset = 0

#     for fid in top_ids:
#         face = shape.Faces[fid]
#         surface = face.Surface

#         # --- Tessellate the face ---
#         # Note: Depending on your FreeCAD version, Standard might be preferred over meshFromShape
#         # for better control, but sticking to your current logic:
#         mesh = MeshPart.meshFromShape(
#             Shape=face,
#             LinearDeflection=tessellation,
#             AngularDeflection=0.523,
#             Relative=False,
#         )

#         # Extract raw vertices
#         raw_verts = [FreeCAD.Vector(v.x, v.y, v.z) for v in mesh.Topology[0]]
#         new_verts = []

#         # --- Per-Vertex Normal Calculation ---
#         for v in raw_verts:
#             try:
#                 # 1. Get UV coordinates on the surface for this 3D point
#                 # This projects the mesh vertex onto the mathematical surface
#                 u, v_param = surface.parameter(v)
                
#                 # 2. Compute the normal at these UV coordinates
#                 n = surface.normal(u, v_param)
                
#                 # 3. Handle orientation (Face normal vs Surface normal)
#                 # FreeCAD Faces can be reversed relative to their geometric Surface
#                 if face.Orientation == 'Reversed':
#                     n.multiply(-1.0)
                
#                 # 4. Normalize (just to be safe, though .normal() usually is)
#                 n.normalize()

#                 # 5. Displace vertex
#                 displaced_v = v.add(n.multiply(offset_dist))
#                 new_verts.append([displaced_v.x, displaced_v.y, displaced_v.z])

#             except Exception as e:
#                 # Fallback if projection fails: keep original position
#                 new_verts.append([v.x, v.y, v.z])

#         # Convert to numpy
#         verts_np = np.array(new_verts)

#         # --- Collect Faces ---
#         for tri in mesh.Topology[1]:
#             # Filter for triangles (FreeCAD mesh usually guarantees this, but safety first)
#             if len(tri) == 3:
#                 all_faces.append([
#                     3,
#                     tri[0] + vert_offset,
#                     tri[1] + vert_offset,
#                     tri[2] + vert_offset,
#                 ])

#         all_verts.append(verts_np)
#         vert_offset += verts_np.shape[0]

#     # Combine all into one PolyData object
#     if not all_verts:
#         return pv.PolyData(), top_ids

#     poly = pv.PolyData(np.vstack(all_verts), np.array(all_faces).flatten())
#     return poly, top_ids