import os, sys
import numpy as np
import pyvista as pv
#from generate_views import render_mesh_views
from tqdm import tqdm
freecad_base = '/data/1bali/miniforge3/envs/multi_view_3DQA'#os.environ["CONDA_PREFIX"]

# Append FreeCAD's Python library paths
sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)

# Now you can import FreeCAD normally
import FreeCAD
import Part
import Import
import Mesh
import MeshPart

## Detect circle in the 

def concave_edges_in_shape(step_file_path):    
    shape = Part.read(step_file_path)

    if shape.ShapeType == "Compound":
        subshapes = shape.childShapes()
        print(f"Total subshapes: {len(subshapes)}")

        solids = [s for s in subshapes if s.ShapeType == "Solid"]
        print(f"Solids found: {len(solids)}")
        shape = solids[0]
    # else:
    #     print(f"Single shape of type: {shape.ShapeType}")
    
    concave_edges, circular_edges = [], []
    for edge in shape.Edges:
        adjacent_faces = shape.ancestorsOfType(edge, Part.Face) #[f for f in shape.Faces if edge in f.Edges]

        if len(adjacent_faces)==2:
            if len(edge.Vertexes) == 2:
                p = (edge.Vertexes[0].Point + edge.Vertexes[1].Point) * 0.5

                normals = []

                eps = 0.4
                ## Orient normals consistently
                for face in adjacent_faces:
                    (u,v) = face.Surface.parameter(p)
                    normal = face.normalAt(u,v)
                    if shape.Solids[0].isInside(p + normal * eps, 1e-7, True):
                        normal = -normal
                    normals.append(normal)

                if shape.Solids[0].isInside((normals[0] + normals[1]) * eps + p, 1e-7, True):
                    if (edge.Vertexes[0].Point, edge.Vertexes[1].Point) not in concave_edges and (edge.Vertexes[1].Point, edge.Vertexes[0].Point) not in concave_edges:
                        print("Concave edge detected:", edge.Vertexes[0].Point," ",edge.Vertexes[1].Point)
                        concave_edges.append((edge.Vertexes[0].Point, edge.Vertexes[1].Point))
            else:
                curve = edge.Curve
                theta_rad = np.random.uniform(0, 2 * np.pi)
                dir1 = curve.Axis.cross(FreeCAD.Vector(0, 1, 0)).normalize()
                dir2 = dir1.cross(curve.Axis).normalize()

                ## Check if the circular edge is convex, inside the circle there should be material
                mark_edge = False
                for r in np.linspace(0, curve.Radius * 0.9, num=5):
                    center = curve.Center
                    if shape.Solids[0].isInside(center + (dir1 * np.cos(theta_rad) + dir2 * np.sin(theta_rad)) * r, 1e-7, True):
                        mark_edge = True
                        break
                if mark_edge: #store the edge anyway
                    circular_edges.append((curve.Radius, curve.Center, curve.Axis))
                    print("Circular edge detected at center:", curve.Center, "with radius:", curve.Radius, "wuth axis:", curve.Axis)
    return concave_edges, circular_edges

def ortho_score(point_of_origin, view_dir, concave_edges, circular_edges, holes_and_fillets, step_file):
    """
    Compute the orthographic view score based on visibility of concave edges.
    view: (elevation, azimuth) in degrees
    concave_edges, circular_edges: list of concave edges as returned by concave_edges_in_shape
    holes_and_fillets: list of holes and fillet surfaces as returned by return_holes_and_fillets
    """
    point_of_origin = FreeCAD.Vector(point_of_origin[0], point_of_origin[1], point_of_origin[2])
    view_dir = FreeCAD.Vector(view_dir[0], view_dir[1], view_dir[2]).normalize()
    shape = Part.read(step_file)
    score = 0
    for edge in concave_edges:
        p1, p2 = edge
        edge_vector = p2 - p1

        if np.linalg.norm(np.array(point_of_origin) - np.array(p1)) < np.linalg.norm(np.array(point_of_origin) - np.array(p2)):
            if shape.Solids[0].isInside(p1 - view_dir * 0.1, 1e-7, True):
                continue # Edge is occluded
        else:
            if shape.Solids[0].isInside(p2 - view_dir * 0.1, 1e-7, True):
                continue # Edge is occluded

        edge_dir = edge_vector.normalize()

        # Check if edge is along the view direction
        if abs(edge_dir.dot(view_dir)) > 0.95:
            score += 1
    
    for circ_surface in holes_and_fillets:
        
        if abs(circ_surface['axis'].dot(view_dir)) > 0.95:
            ## Check for occlusion
            center = circ_surface['center']
            L = 1e6
            p0 = center - view_dir * L
            p1 = center + view_dir * L
            ray = Part.makeLine(p0, p1)

            intersection = shape.common(ray)
            if len(intersection.Vertexes) > 0: continue # not a hole at all
            # first_hit = [v.Point for v in intersection.Vertexes][0]

            # if np.linalg.norm(np.array(first_hit) - np.array(p0)) < np.linalg.norm(np.array(center) - np.array(p0)):
            #     continue # circular hole is occluded
            else: score += 1


    for circle in circular_edges:
        radius, center, axis = circle
        axis_dir = axis.normalize()
        
        # SAFEGUARD: If axis_dir is parallel to (0,1,0), cross product becomes zero.
        # Use a fallback vector to be safe.
        up_vec = FreeCAD.Vector(0, 1, 0)
        if abs(axis_dir.dot(up_vec)) > 0.99:
            up_vec = FreeCAD.Vector(1, 0, 0)
            
        dir1 = axis_dir.cross(up_vec).normalize()
        dir2 = dir1.cross(axis_dir).normalize()

        if abs(axis_dir.dot(view_dir)) > 0.95:
            # 1. Cast Ray to the CENTER
            L = 1e6
            p0 = center - view_dir * L # Camera
            p1 = center + view_dir * L # Far behind
            ray_center = Part.makeLine(p0, p1)

            # Inside your circular_edges loop, after defining the center ray...
            int_center = shape.common(ray_center)

            # --- START OF FIX ---

            # Case 1: The ray doesn't hit anything at all. This means it's definitely visible.
            if not int_center.Vertexes:
                score += 1
                continue # Go to the next circle

            # Sort hits to guarantee we get the FIRST one closest to the camera
            hits_center = [v.Point for v in int_center.Vertexes]
            first_hit_center = min(hits_center, key=lambda p: (p - p0).Length)

            dist_hit_center = (first_hit_center - p0).Length
            dist_actual_center = (center - p0).Length

            # Case 2: The ray hits something. Check if the hit is almost exactly AT the center.
            # If the hit is very close to the center, it means the first thing the ray saw
            # was the bottom of the hole, so it's visible.
            if abs(dist_hit_center - dist_actual_center) < 1e-4: # Tolerance for floating point
                score += 1
                continue # It's visible, move on

            # Case 3: The ray hits something SIGNIFICANTLY IN FRONT of the center.
            # This is the original logic for a protruding boss.
            protrusion_length = (first_hit_center - center).Length
            if protrusion_length > 0.1: # Threshold for a meaningful protrusion
                
                # (Your existing logic to check if the edge is visible...)
                theta_rad = np.random.uniform(0, 2 * np.pi)
                edge_point = center + dir1 * (radius * np.cos(theta_rad)) + dir2 * (radius * np.sin(theta_rad))
                p0_edge = edge_point - view_dir * L
                ray_edge = Part.makeLine(p0_edge, p0_edge + view_dir * 2*L) # Corrected ray creation
                
                int_edge = shape.common(ray_edge)
                if int_edge.Vertexes:
                    hits_edge = [v.Point for v in int_edge.Vertexes]
                    first_hit_edge = min(hits_edge, key=lambda p: (p - p0_edge).Length)
                    
                    dist_hit_edge = (first_hit_edge - p0_edge).Length
                    dist_actual_edge = (edge_point - p0_edge).Length
                    
                    if dist_hit_edge >= dist_actual_edge - 1e-4:
                        score += 1
            # --- END OF FIX ---
    return score

def nearest_face(shape, point):
    min_dist = 1e9
    closest_face = None

    for face in shape.Faces:
        dist = face.distToShape(Part.Vertex(point))[0]
        if dist < min_dist:
            min_dist = dist
            closest_face = face

    return closest_face, min_dist

def is_internal_face(shape, face, eps=0.5):
    umin, umax, vmin, vmax = face.ParameterRange
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)

    p = face.valueAt(u, v)
    n = face.normalAt(u, v)

    return shape.Solids[0].isInside(p + n * eps, 1e-7, True)

def is_cylindrical_hole_face(face, solid, eps=1e-4):
    surf = face.Surface
    if not isinstance(surf, Part.Cylinder):
        return False

    # Sample midpoint
    umin, umax, vmin, vmax = face.ParameterRange
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)

    p = face.valueAt(u, v)
    n = face.normalAt(u, v)

    inside_plus  = solid.isInside(p + n * eps,  1e-7, True)
    inside_minus = solid.isInside(p - n * eps,  1e-7, True)

    # Hole condition:
    # exactly one side is inside solid
    return inside_plus ^ inside_minus

def normalize(v):
    v = np.array([v.x, v.y, v.z])
    return tuple((v / np.linalg.norm(v)).round(6))

def round_vec(v, tol=1e-4):
    return tuple(np.round([v.x, v.y, v.z], int(-np.log10(tol))))

def hole_key(cyl):
    axis_dir = normalize(cyl.Axis)
    axis_pos = round_vec(cyl.Center)
    radius   = round(cyl.Radius, 4)

    return (radius, axis_dir, axis_pos)

def is_true_hole_cylinder(face, solid, nsamples=6, eps=1e-4):
    cyl = face.Surface
    axis = cyl.Axis.normalize()

    # sample a stable point on surface
    umin, umax, vmin, vmax = face.ParameterRange
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)
    p = face.valueAt(u, v)

    # radial direction
    vcp = p - cyl.Center
    radial = (vcp - axis * vcp.dot(axis)).normalize()

    unit_vcp = vcp.normalize()
    vcp_len = vcp
    R = cyl.Radius

    # sample points from center outward (avoid exact boundaries)
    for t in np.linspace(0.1, 0.9, nsamples):
        q = cyl.Center + unit_vcp * t * vcp.Length#radial * (t * R)

        if solid.isInside(q, 1e-7, True):
            return False   # material found → not a hole

    return True

def detect_holes_and_fillets(step_file_path):
    shape = Part.read(step_file_path)
    
    if shape.ShapeType == "Compound":
        subshapes = shape.childShapes()
        print(f"Total subshapes: {len(subshapes)}")

        solids = [s for s in subshapes if s.ShapeType == "Solid"]
        print(f"Solids found: {len(solids)}")
        shape = solids[0]
    # else:
    #     print(f"Single shape of type: {shape.ShapeType}")

    solid = shape.Solids[0]

    holes = []
    seen = set()
    for face in shape.Faces:
        if not isinstance(face.Surface, Part.Cylinder):
            continue

        if not is_cylindrical_hole_face(face, solid):
            continue

        cyl = face.Surface

        if is_true_hole_cylinder(face, solid, nsamples=6):  

            key = hole_key(cyl)
            if key not in seen:
                seen.add(key)
                holes.append({
                    "surf": cyl,
                    "radius": cyl.Radius,
                    "axis": cyl.Axis,
                    "center": cyl.Center
                })

                print(f"Hole detected:",
                "R =", cyl.Radius,
                "Axis =", cyl.Axis)
                

    return holes

def is_fillet_face(face, solid, eps=1e-4):
    surf = face.Surface

    if not isinstance(surf, (Part.Cylinder, Part.Toroid)):
        return False

    umin, umax, vmin, vmax = face.ParameterRange
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)

    p = face.valueAt(u, v)
    n = face.normalAt(u, v)

    # Convex test: normal should point OUT of solid
    return not solid.isInside(p + n * eps, 1e-7, True)

def detect_fillets(step_file_path, eps=1e-1):
    shape = Part.read(step_file_path)
    solid = shape.Solids[0]

    fillets = []

    for edge in shape.Edges:

        # 1️⃣ Edge must be circular
        
        try:
            if isinstance(edge.Curve, Part.Circle):
                pass
            else:
                continue
        except: continue

        # 2️⃣ Must connect exactly two faces
        faces = shape.ancestorsOfType(edge, Part.Face)
        if len(faces) != 2:
            continue

        circle = edge.Curve
        center = circle.Center
        axis   = circle.Axis.normalize()
        radius = circle.Radius

        # 3️⃣ Pick a stable point on the edge
        p = edge.valueAt(0.5 * (edge.FirstParameter + edge.LastParameter))

        # 4️⃣ Build radial direction (orthogonal to axis)
        radial = (p - center).normalize()
        #radial = (v - axis * v.dot(axis)).normalize()

        # 5️⃣ Probe slightly inward and outward
        p_out = p - radial * eps
        p_in = p + radial * eps

        inside_in  = solid.isInside(p_in,  1e-7, True)
        inside_out = solid.isInside(p_out, 1e-7, True)

        # 6️⃣ Fillet condition:
        #     material inside, air outside
        if inside_in and not inside_out:
            fillets.append({
                "radius": radius,
                "axis": axis,
                "center": center,
                "edge": edge,
                "faces": faces
            })

            print("Fillet detected:",
                  "R =", radius,
                  "Axis =", axis,
                  "Center =", center)

    return fillets


#def perpendicular_circular_edges(step_file, load_dir = FreeCAD.Vector(0,-1,0)):


# serial
# if __name__ == "__main__":
#     #step_file = "/data/1bali/Other_LLM_projects/multi_view_3DQA/custom_parts/Electrical_Parts_Servos_SG-90_Servo-sg90/Electrical_Parts_Servos_SG-90_Servo-sg90.step"  # Replace with your STEP file path
#     step_file = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset/Electronics_Parts_Optical_SHARP_2Y0A02_SHARP-2Y0A02/Electronics_Parts_Optical_SHARP_2Y0A02_SHARP-2Y0A02.step"
#     step_file = "/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset/Electrical_Parts_Servos_SG-90_Servo-sg90/Electrical_Parts_Servos_SG-90_Servo-sg90.step"
    
#     #dataset_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset'
#     dataset_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset_additional'
#     cad_folders = os.listdir(dataset_path)

#     for cad_folder in tqdm(cad_folders,
#                        total=len(cad_folders),
#                        desc="Processing CAD folders",
#                        unit="folder"):
        
#         for file_step in os.listdir(f"{dataset_path}/{cad_folder}"):
#             if file_step.endswith(".step"):
#                 step_file = f"{dataset_path}/{cad_folder}/{file_step}"
        
#         # dEBUG 
#         # if step_file not in ['/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset/Mechanical_Parts_Pulleys_GT2Pulley-V3/Mechanical_Parts_Pulleys_GT2Pulley-V3.step']: continue
        
#         ## sometimes stp files are more accurate than step files
#         alt_step_file = step_file.replace("step", "stp")
#         if os.path.exists(alt_step_file):
            
#             step_file = alt_step_file

#         try:
#             mesh_file = step_file.replace('.step', '.obj').replace('.stp', '.obj')  # Assuming the mesh file has the same name with .stl extension
#             mesh = pv.read(mesh_file)
#         except:
#             mesh_file = "/".join(step_file.split('/')[:-1]) + f'/renders_pyvista/{step_file.split("/")[-1].replace(".step",".obj").replace(".stp",".obj")}'
#             mesh = pv.read(mesh_file)
        
#         bounds = mesh.bounds
#         center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
#         zoom = 1 
#         radius = np.linalg.norm([
#             bounds[1]-bounds[0],
#             bounds[3]-bounds[2],
#             bounds[5]-bounds[4]
#         ])/2 * zoom

#         concave_edges, circular_edges = concave_edges_in_shape(step_file)
#         holes_and_fillets = detect_holes_and_fillets(step_file)
#         fillets = detect_fillets(step_file)
#         holes_and_fillets.extend(fillets)

#         print(f"Step File: {step_file}")
#         print(f"Total concave edges found: {len(concave_edges)}")
#         print(f"Total circular edges found: {len(circular_edges)}")
#         print(f"Total Holes/Fillets found: {len(holes_and_fillets)}")
#         ## Mark it on the CAD geometry
#         # mark_points = []
#         # for edge in concave_edges:
#         #     p1, p2 = edge
#         #     p1 = np.array([p1.x, p1.y, p1.z])
#         #     p2 = np.array([p2.x, p2.y, p2.z])
#         #     # Generate points along the concave edge
#         #     edge_points = p1 + (p2 - p1) * np.linspace(0, 1, num=15)[:, None]
#         #     mark_points.extend(edge_points)

#         # for circle in circular_edges:
#         #     radius, center, axis = circle
#         #     center = np.array([center.x, center.y, center.z])
#         #     axis_dir = axis.normalize()
#         #     # Generate points on the circular edge
#         #     for angle in np.linspace(0, 2 * np.pi, num=30):
                
#         #         dir1 = axis_dir.cross(FreeCAD.Vector(0, 1, 0)).normalize()
#         #         dir2 = dir1.cross(axis_dir).normalize()

#         #         dir1 = np.array([dir1.x, dir1.y, dir1.z])
#         #         dir2 = np.array([dir2.x, dir2.y, dir2.z])
                
#         #         point_on_circle = center + radius * (dir1 * np.cos(angle) + dir2 * np.sin(angle))
#         #         mark_points.append(point_on_circle)
        

        
#         # # for circ_surf in holes_and_fillets:

#         # CAD_folder = '/'.join(step_file.split('/')[:-1])
#         # render_mesh_views(mesh_file, output_dir=f'{CAD_folder}/renders_pyvista_with_points', n_azimuth=12, n_elevation=9, orthographic=False, points_3d=mark_points)
#         # ## Compute optimal views that maximize visibility of concave edges
#         n_elevation = 1
#         n_azimuth = 12
#         if isinstance(n_elevation, list):
#             elevations = n_elevation
#         else:
#             elevations = [-90 + (180 /(n_elevation+1)) * e for e in range(1, n_elevation+1)]

#         if isinstance(n_azimuth, list):
#             azimuths = n_azimuth
#         else:
#             azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]

#         ortho_score_dict = {}
#         for elevation in elevations:
#             for azimuth in azimuths:
#                 # Convert spherical coordinates to Cartesian camera position
#                 view_x = np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
#                 view_y = np.sin(np.radians(elevation))
#                 view_z = - np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
#                 view_dir = np.array([view_x, view_y, view_z])
#                 cam_pos = center - radius * view_dir
#                 score = ortho_score(cam_pos, view_dir, concave_edges, circular_edges, holes_and_fillets, step_file)

#                 print(f"View (elev: {elevation}, azim: {azimuth}) - Ortho Score: {score}")
#                 ortho_score_dict[f'view_e{elevation}_a{azimuth}'] = score
        
#         sorted_d = dict(sorted(ortho_score_dict.items(), key=lambda item: item[1], reverse=True))
#         with open(f"{dataset_path}/{cad_folder}/ortho_views.txt", 'w') as fout:
#             for item in sorted_d.items():
#                 fout.write(f'{item[0]}: {item[1]}\n')
        
#         print(f'Processed {step_file}')

## parallel

import os
import numpy as np
import pyvista as pv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        concave_edges, circular_edges = concave_edges_in_shape(step_file)
        holes_and_fillets = detect_holes_and_fillets(step_file)
        fillets = detect_fillets(step_file)
        holes_and_fillets.extend(fillets)

        # --- view scoring ---
        n_elevation = 1
        n_azimuth = 12

        elevations = [-90 + (180 /(n_elevation+1)) * e for e in range(1, n_elevation+1)]
        azimuths = [(360 / n_azimuth) * a for a in range(n_azimuth)]

        ortho_score_dict = {}

        for elevation in elevations:
            for azimuth in azimuths:
                view_x = np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
                view_y = np.sin(np.radians(elevation))
                view_z = - np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))

                view_dir = np.array([view_x, view_y, view_z])
                cam_pos = center - radius * view_dir

                score = ortho_score(cam_pos, view_dir,
                                    concave_edges,
                                    circular_edges,
                                    holes_and_fillets,
                                    step_file)
                
                print(f"File: {step_file} View (elev: {elevation}, azim: {azimuth}) - Ortho Score: {score}")

                ortho_score_dict[f'view_e{elevation}_a{azimuth}'] = score

        # --- save ---
        sorted_d = dict(sorted(ortho_score_dict.items(),
                               key=lambda item: item[1],
                               reverse=True))

        with open(os.path.join(folder_path, "ortho_views.txt"), 'w') as fout:
            for k, v in sorted_d.items():
                fout.write(f'{k}: {v}\n')

        return f"[DONE] {cad_folder}"

    except Exception as e:
        return f"[ERROR] {cad_folder}: {str(e)}"


# ---------------- MAIN ----------------
if __name__ == "__main__":

    dataset_path = '/data/1bali/Other_LLM_projects/multi_view_3DQA/ortho_views/dataset_additional_test'
    cad_folders = os.listdir(dataset_path)

    tasks = [(cad_folder, dataset_path) for cad_folder in cad_folders]

    # ⚠️ IMPORTANT: tune this
    max_workers = 16   # start with 4–8 for FreeCAD

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_cad, task) for task in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            print(f.result())
