import os, sys
import numpy as np
import pyvista as pv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

freecad_base = '/data/1bali/miniforge3/envs/multi_view_3DQA'

# Append FreeCAD's Python library paths
sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)

import FreeCAD
import Part

# =====================================================================
# 1. GEOMETRY DETECTION FUNCTIONS
# =====================================================================

def concave_edges_in_shape(step_file_path):    
    full_shape = Part.read(step_file_path)
    
    # ENFORCE: Only consider the first solid body
    if len(full_shape.Solids) > 0:
        shape = full_shape.Solids[0]
    else:
        shape = full_shape # Fallback
        
    solid = shape.Solids[0]
    
    concave_edges = []
    arbitrary_closed_contours = []
    
    # =======================================================
    # 1. DETECT ALL INDIVIDUAL CONCAVE EDGES (Both open and closed)
    # =======================================================
    for edge in shape.Edges:
        adjacent_faces = shape.ancestorsOfType(edge, Part.Face)

        if len(adjacent_faces) == 2:
            p = edge.valueAt(0.5 * (edge.FirstParameter + edge.LastParameter))
            normals = []
            eps = 1e-3 
            
            for face in adjacent_faces:
                (u,v) = face.Surface.parameter(p)
                normal = face.normalAt(u,v).normalize()
                if solid.isInside(p + normal * eps, 1e-7, True):
                    normal = -normal
                normals.append(normal)

            test_vec = normals[0] - normals[1]
            
            if test_vec.Length > 1e-4:
                test_vec = test_vec.normalize()
                
                if solid.isInside(p + test_vec * eps, 1e-7, True):
                    # This edge is geometrically concave. Now we categorize it.
                    if len(edge.Vertexes) >= 2:
                        p1, p2 = edge.Vertexes[0].Point, edge.Vertexes[-1].Point
                    else:
                        p1, p2 = edge.valueAt(edge.FirstParameter), edge.valueAt(edge.LastParameter)
                        
                    edge_vec = p2 - p1
                    
                    if edge_vec.Length < 1e-4 or edge.isClosed():
                        # It's a closed concave loop (e.g., bottom of a blind hole)
                        center = edge.BoundBox.Center
                        equiv_radius = edge.BoundBox.DiagonalLength / 2.0
                        axis = (normals[0] + normals[1]).normalize() 
                        arbitrary_closed_contours.append((equiv_radius, center, axis))
                    else:
                        # It's an open concave edge (e.g., inner corner of L-bracket)
                        if (p1, p2) not in concave_edges and (p2, p1) not in concave_edges:
                            concave_edges.append((p1, p2))

    # =======================================================
    # 2. DETECT PREDOMINANTLY CONCAVE INNER CONTOURS
    # =======================================================
    for face in solid.Faces:
        if not isinstance(face.Surface, Part.Plane):
            continue
            
        # if len(face.Wires) <= 1:
        #     continue

        face_normal = face.normalAt(*face.Surface.parameter(face.CenterOfMass)).normalize()
        if solid.isInside(face.CenterOfMass + face_normal * 1e-3, 1e-7, True):
            face_normal = -face_normal

        outer_wire = face.OuterWire
        
        for wire in face.Wires:
            # if wire.isSame(outer_wire):
            #     continue

            # This is an inner wire. We must traverse it completely.
            concave_edge_count = 0
            
            # --- TRAVERSE THE ENTIRE WIRE, NO BREAKING ---
            for edge in wire.Edges:
                adj_faces = shape.ancestorsOfType(edge, Part.Face)
                if len(adj_faces) != 2: continue
                
                p = edge.valueAt(0.5 * (edge.FirstParameter + edge.LastParameter))
                normals = []
                eps = 1e-3
                for f in adj_faces:
                    (u,v) = f.Surface.parameter(p)
                    n = f.normalAt(u,v).normalize()
                    if solid.isInside(p + n * eps, 1e-7, True):
                        n = -n
                    normals.append(n)

                test_vec = normals[0] - normals[1]
                if test_vec.Length > 1e-4:
                    test_vec = test_vec.normalize()
                    if solid.isInside(p + test_vec * eps, 1e-7, True):
                        # This edge is concave. Increment the counter.
                        concave_edge_count += 1
            
            # --- FINAL DECISION BASED ON THE VOTE ---
            # If at least half the edges are concave, we classify it as a concave feature.
            # Using >= allows for features with equal concave/convex edges (like a slot with rounded ends) to be counted.
            if len(wire.Edges) > 0 and concave_edge_count == len(wire.Edges):
                center = wire.CenterOfMass
                equiv_radius = wire.BoundBox.DiagonalLength / 2.0
                arbitrary_closed_contours.append((equiv_radius, center, face_normal))

    return concave_edges, arbitrary_closed_contours


def detect_arbitrary_concave_faces(step_file_path):
    full_shape = Part.read(step_file_path)
    if len(full_shape.Solids) > 0:
        shape = full_shape.Solids[0]
    else:
        shape = full_shape

    concave_faces = []
    for face in shape.Faces:
        if isinstance(face.Surface, Part.Plane): continue

        # 1. Get parametric center p and outward normal n
        umin, umax, vmin, vmax = face.ParameterRange
        u, v = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
        p = face.valueAt(u, v)
        normal = face.normalAt(u, v).normalize()
        
        if shape.isInside(p + normal * 1e-3, 1e-7, True):
            normal = -normal
            
        # 2. Concavity check
        is_concave = False
        for edge in face.Edges:
            q = edge.valueAt(0.5 * (edge.FirstParameter + edge.LastParameter))
            if (q - p).dot(normal) > 1e-3: 
                is_concave = True
                break
                
        if is_concave:
            # --- START OF CENTER & AXIS FIX ---
            feature_axis = None
            feature_center = None
            radius = 1.0 # Default fallback shift

            # A. If it's a Cylinder or Cone, the center is on the mathematical axis
            if hasattr(face.Surface, 'Axis') and hasattr(face.Surface, 'Center'):
                surf = face.Surface
                feature_axis = surf.Axis.normalize()
                # Project surface point p onto the mathematical axis line
                v = p - surf.Center
                dist_along_axis = v.dot(feature_axis)
                feature_center = surf.Center + (feature_axis * dist_along_axis)
                
                if hasattr(surf, 'Radius'):
                    radius = surf.Radius
            
            # B. Fallback: Use Curvature to find the "Air Center"
            if feature_center is None:
                try:
                    # curvatureAt returns (k1, k2). For a fillet, one is 1/R, one is 0.
                    curvatures = face.curvatureAt(u, v)
                    max_curv = max(abs(curvatures[0]), abs(curvatures[1]))
                    if max_curv > 1e-5:
                        radius = 1.0 / max_curv
                    # Move from surface p, OPPOSITE to outward normal, into the air
                    feature_center = p - (normal * radius)
                except:
                    feature_center = p - (normal * 5.0) # Move 5mm into air as guess

            # Ensure we have an axis for scoring
            if feature_axis is None:
                # Fallback axis detection from longest edge
                for edge in face.Edges:
                    if isinstance(edge.Curve, Part.Line):
                        p1, p2 = edge.valueAt(edge.FirstParameter), edge.valueAt(edge.LastParameter)
                        feature_axis = (p2 - p1).normalize()
                        break
                if feature_axis is None: feature_axis = normal # Last resort

            # --- END OF FIX ---
                
            concave_faces.append({
                "center": feature_center, # Now sits in the AIR
                "axis": feature_axis, 
                "radius": radius
            })

    return concave_faces


# =====================================================================
# 2. SCORING ALGORITHM
# =====================================================================

def ortho_score(point_of_origin, view_dir, concave_edges, closed_contours, concave_faces, step_file):
    point_of_origin = FreeCAD.Vector(point_of_origin[0], point_of_origin[1], point_of_origin[2])
    view_dir = FreeCAD.Vector(view_dir[0], view_dir[1], view_dir[2]).normalize()
    full_shape = Part.read(step_file)
    # ENFORCE: Only consider the first solid body
    if len(full_shape.Solids) > 0:
        shape = full_shape.Solids[0]
    else:
        shape = full_shape
    concave_edges_score, concave_faces_score, closed_contours_score = 0, 0, 0
    
    # 1. Score standard open concave edges
    for edge in concave_edges:
        p1, p2 = edge
        edge_vector = p2 - p1

        if np.linalg.norm(np.array(point_of_origin) - np.array(p1)) < np.linalg.norm(np.array(point_of_origin) - np.array(p2)):
            if shape.Solids[0].isInside(p1 - view_dir * 0.1, 1e-7, True):
                continue
        else:
            if shape.Solids[0].isInside(p2 - view_dir * 0.1, 1e-7, True):
                continue

        edge_dir = edge_vector.normalize()
        if abs(edge_dir.dot(view_dir)) > 0.99:
            concave_edges_score += 1
            
    # 2. Score concave faces (fillets, grooves, swept blends)
    for face_dict in concave_faces:
        if abs(face_dict['axis'].dot(view_dir)) > 0.99:
            center = face_dict['center']
            L = 1e6
            p0 = center - view_dir * L
            p1 = center + view_dir * L
            ray = Part.makeLine(p0, p1)

            intersection = shape.common(ray)
            if len(intersection.Vertexes) == 0: 
                concave_faces_score += 1

    # 3. Score closed contours (holes, slots, pockets, bosses, and circular blind holes)
    for contour in closed_contours:
        radius, center, axis = contour
        axis_dir = axis.normalize()
        
        up_vec = FreeCAD.Vector(0, 1, 0)
        if abs(axis_dir.dot(up_vec)) > 0.99:
            up_vec = FreeCAD.Vector(1, 0, 0)
            
        dir1 = axis_dir.cross(up_vec).normalize()
        dir2 = dir1.cross(axis_dir).normalize()

        if abs(axis_dir.dot(view_dir)) > 0.99:
            L = 1e6
            p0 = center - view_dir * L
            p1 = center + view_dir * L
            ray_center = Part.makeLine(p0, p1)

            int_center = shape.common(ray_center)

            if not int_center.Vertexes:
                closed_contours_score += 1
                continue 

            hits_center = [v.Point for v in int_center.Vertexes]
            first_hit_center = min(hits_center, key=lambda p: (p - p0).Length)

            dist_hit_center = (first_hit_center - p0).Length
            dist_actual_center = (center - p0).Length

            if abs(dist_hit_center - dist_actual_center) < 1e-4:
                closed_contours_score += 1
                continue 

            protrusion_length = (first_hit_center - center).Length
            if protrusion_length > 0.1: 
                theta_rad = np.random.uniform(0, 2 * np.pi)
                edge_point = center + dir1 * (radius * np.cos(theta_rad)) + dir2 * (radius * np.sin(theta_rad))
                p0_edge = edge_point - view_dir * L
                ray_edge = Part.makeLine(p0_edge, p0_edge + view_dir * 2*L) 
                
                int_edge = shape.common(ray_edge)
                if int_edge.Vertexes:
                    hits_edge = [v.Point for v in int_edge.Vertexes]
                    first_hit_edge = min(hits_edge, key=lambda p: (p - p0_edge).Length)
                    
                    dist_hit_edge = (first_hit_edge - p0_edge).Length
                    dist_actual_edge = (edge_point - p0_edge).Length
                    
                    if dist_hit_edge >= dist_actual_edge - 1e-4:
                        closed_contours_score += 1

    return concave_edges_score + concave_faces_score + closed_contours_score


# =====================================================================
# 3. PARALLEL PROCESSING PIPELINE
# =====================================================================

def process_single_cad(args):
    cad_folder, dataset_path = args

    try:
        import sys
        freecad_base = '/data/1bali/miniforge3/envs/multi_view_3DQA'
        sys.path.append(os.path.join(freecad_base, "lib"))
        sys.path.append(os.path.join(freecad_base, "Mod"))

        import FreeCAD
        import Part

        # --- locate STEP file ---
        step_file = None
        folder_path = os.path.join(dataset_path, cad_folder)

        for file_step in os.listdir(folder_path):
            if file_step.endswith(".step") or file_step.endswith(".stp"):
                step_file = os.path.join(folder_path, file_step)
                break

        if step_file is None:
            return f"[SKIP] No STEP in {cad_folder}"

        alt_step_file = step_file.replace("step", "stp")
        if os.path.exists(alt_step_file):
            step_file = alt_step_file

        # --- load mesh ---
        try:
            mesh_file = step_file.replace('.step', '.obj').replace('.stp', '.obj')
            mesh = pv.read(mesh_file)
        except:
            mesh_file = os.path.join(folder_path, "renders_pyvista",
                                     os.path.basename(step_file).replace(".step", ".obj").replace(".stp", ".obj"))
            mesh = pv.read(mesh_file)

        bounds = mesh.bounds
        center = np.array([(bounds[0]+bounds[1])/2,
                           (bounds[2]+bounds[3])/2,
                           (bounds[4]+bounds[5])/2])

        radius = np.linalg.norm([
            bounds[1]-bounds[0],
            bounds[3]-bounds[2],
            bounds[5]-bounds[4]
        ]) / 2

        # --- geometry analysis ---
        concave_edges, closed_contours = concave_edges_in_shape(step_file)
        concave_faces = detect_arbitrary_concave_faces(step_file)

        # --- view scoring ---
        n_elevation = [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90]
        n_azimuth = 12

        #elevations = [-90 + (180 /(n_elevation+1)) * e for e in range(1, n_elevation+1)]
        elevations = n_elevation
        azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]

        ortho_score_dict = {}

        for elevation in elevations:
            for azimuth in azimuths:
                view_x = - np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
                view_y = - np.sin(np.radians(elevation))
                view_z = - np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

                view_dir = np.array([view_x, view_y, view_z])
                cam_pos = center - radius * view_dir

                score = ortho_score(cam_pos, view_dir,
                                    concave_edges,
                                    closed_contours,
                                    concave_faces,
                                    step_file)
                
                # Uncomment to print scores per file
                # print(f"File: {step_file} View (elev: {elevation}, azim: {azimuth}) - Ortho Score: {score}")

                ortho_score_dict[f'view_e{elevation}_a{azimuth}'] = score

                if elevation in [-90, 90]:  break  # No need to compute all azimuths for top/bottom views

        # --- save ---
        sorted_d = dict(sorted(ortho_score_dict.items(),
                               key=lambda item: item[1],
                               reverse=True))

        consider_flag = 0
        for k, v in sorted_d.items():
            if v > 0:
                consider_flag=1
                break
        
        if consider_flag == 0:
            return f"[SKIP] {cad_folder} has no concave features."
        
        with open(os.path.join(folder_path, "ortho_views.txt"), 'w') as fout:
            for k, v in sorted_d.items():
                fout.write(f'{k}: {v}\n')

        return f"[DONE] {cad_folder}"

    except Exception as e:
        return f"[ERROR] {cad_folder}: {str(e)}"

import re # Add this to your imports

def parse_log_file(log_path):
    """
    Reads a log file and returns a set of CAD folder names that have already been
    processed (DONE, SKIP, TIMEOUT, ERROR).
    """
    processed_folders = set()
    if not os.path.exists(log_path):
        return processed_folders  # Return empty set if log file doesn't exist

    # Regex to find a status in brackets followed by a CAD folder name (e.g., "[DONE] 00500013")
    # It looks for a word in [], followed by spaces, then a sequence of digits.
    log_pattern = re.compile(r"\[(DONE|SKIP|TIMEOUT|ERROR)\]\s+([0-9a-zA-Z_-]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                cad_folder = match.group(2)
                processed_folders.add(cad_folder)
    return processed_folders

# ---------------- MAIN ----------------

## Parallel processing with timeout handling
import concurrent.futures # Needed for the TimeoutError exception

# ---------------- MAIN ----------------

## Parallel processing with timeout handling
import concurrent.futures # Needed for the TimeoutError exception

if __name__ == "__main__":

    dataset_path = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2'
    log_file_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/GReFEM/generate_orthoviews.log' # Define the path to your log file

    # --- START OF NEW LOGIC ---
    print(f"Reading existing log file: {log_file_path}")
    already_processed = parse_log_file(log_file_path)
    print(f"Found {len(already_processed)} folders already processed. They will be skipped.")

    all_cad_folders = os.listdir(dataset_path)
    # Filter the list to only include folders that have NOT been processed
    cad_folders_to_process = [f for f in all_cad_folders if f not in already_processed]

    print(f"Total folders to process: {len(cad_folders_to_process)} out of {len(all_cad_folders)}")
    # --- END OF NEW LOGIC ---

    # The rest of the code now works on the filtered list
    tasks = [(cad_folder, dataset_path) for cad_folder in cad_folders_to_process]
    
    # If there are no new tasks, exit gracefully
    if not tasks:
        print("No new CAD folders to process. Exiting.")
        sys.exit()

    max_workers = 30   
    timeout_seconds = 10 

    # IMPORTANT: To resume logging, open the log file in 'append' mode ('a')
    with open(log_file_path, 'a') as log_f, ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cad = {executor.submit(process_single_cad, task): task[0] for task in tasks}

        pbar = tqdm(total=len(tasks), desc="Processing New CADs")

        for future in concurrent.futures.as_completed(future_to_cad):
            cad_name = future_to_cad[future]
            try:
                result = future.result(timeout=timeout_seconds)
                log_f.write(result + '\n') # Write result to the log file
                print(result)
            except concurrent.futures.TimeoutError:
                result_str = f"[TIMEOUT] {cad_name} exceeded {timeout_seconds}s limit."
                log_f.write(result_str + '\n') # Log timeout
                print(result_str)
            except Exception as e:
                result_str = f"[ERROR] {cad_name}: {e}"
                log_f.write(result_str + '\n') # Log error
                print(result_str)
            finally:
                pbar.update(1)
        
        pbar.close()

## Sequential processing (uncomment to use instead of parallel)
# if __name__ == "__main__":
#     dataset_path = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2'
#     cad_folders = os.listdir(dataset_path)

#     for cad_folder in tqdm(cad_folders, desc="Processing CADs"):
#         if cad_folder not in ['00410018']: continue
#         result = process_single_cad((cad_folder, dataset_path))
#         print(result)
