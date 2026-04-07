import sys
import os
import vtk
import gmsh
from tqdm import tqdm
from ufl import Identity
import multiprocessing as mp
from generate_views import render_pyvista_views
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

import signal


from dolfinx import mesh, fem
import numpy as np
import ufl

def run_bc_diagnostics(msh, facet_normals, top_facets, bottom_facets, y_min=None, y_max=None):
    """
    Prints diagnostic info about detected boundary facets and normals.
    Helps verify that BC tagging is correct and consistent.
    """
    geom = msh.geometry.x
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)

    def summarize_facets(facets, label):
        if len(facets) == 0:
            print(f"⚠️  No {label} facets found!")
            return

        y_vals = []
        ny_vals = []
        for f in facets[:min(10, len(facets))]:
            verts = conn.links(f)
            y_mean = geom[verts, 1].mean()
            n_y = facet_normals[f, 1]
            y_vals.append(y_mean)
            ny_vals.append(n_y)

        print(f"---- {label.upper()} FACETS ----")
        print(f"Count: {len(facets)}")
        print(f"Mean y: {np.mean(y_vals):.6f}, range: ({np.min(y_vals):.6f}, {np.max(y_vals):.6f})")
        print(f"Mean normal_y: {np.mean(ny_vals):.3f}, range: ({np.min(ny_vals):.3f}, {np.max(ny_vals):.3f})")
        print(f"Sample y_mean vs n_y pairs: {list(zip(y_vals[:5], ny_vals[:5]))}")
        print("------------------------------")

    print("\n====== BC DIAGNOSTICS ======")
    if y_min is not None and y_max is not None:
        print(f"y_min={y_min:.4f}, y_max={y_max:.4f}")
    print(f"Facet_normals shape: {facet_normals.shape}")
    print(f"Top facets count: {len(top_facets)}, Bottom facets count: {len(bottom_facets)}")

    summarize_facets(top_facets, "top")
    summarize_facets(bottom_facets, "bottom")
    print("====== END BC DIAGNOSTICS ======\n")

## Improve if time permits, currently computes normals for all faces inside mesh!!
def compute_normals(msh):
    """
    Compute approximate outward facet normals for a 3D mesh (triangular faces),
    fully vectorized (no Python loops).
    Works for Dolfinx >= 0.6.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Build connectivity facet -> vertex
    msh.topology.create_connectivity(fdim, 0)
    facet_conn = msh.topology.connectivity(fdim, 0)
    num_facets = msh.topology.index_map(fdim).size_local

    # Build flat index arrays (facet -> [v0,v1,v2])
    offsets = facet_conn.offsets
    indices = facet_conn.array

    # Each facet may have different number of vertices (tri or quad)
    # We'll only use the first 3 vertices per facet to define a plane
    num_facets_valid = np.sum(np.diff(offsets) >= 3)
    v0 = indices[offsets[:-1][np.diff(offsets) >= 3]]
    v1 = indices[offsets[:-1][np.diff(offsets) >= 3] + 1]
    v2 = indices[offsets[:-1][np.diff(offsets) >= 3] + 2]

    coords = msh.geometry.x
    p0, p1, p2 = coords[v0], coords[v1], coords[v2]

    # Compute cross products for all facets in one go
    v1v0 = p1 - p0
    v2v0 = p2 - p0
    n = np.cross(v1v0, v2v0)

    # Normalize safely
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-12
    n[valid] /= norms[valid, np.newaxis]

    # Allocate full array of normals (for all facets)
    normals = np.zeros((num_facets, 3))
    normals[np.diff(offsets) >= 3] = n

    return normals


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def fcstd_to_step(input_fcstd, output_step):
    # Load the FreeCAD document
    doc = FreeCAD.open(input_fcstd)

    # Collect all solid/shape objects
    objs = [obj for obj in doc.Objects if hasattr(obj, "Shape") and not obj.Shape.isNull()]

    if not objs:
        raise RuntimeError("No solid objects found in the FCStd file!")

    # Export to STEP
    Import.export(objs, output_step)

    print(f"✅ Exported {input_fcstd} → {output_step}")

    # Close document
    FreeCAD.closeDocument(doc.Name)

## Debug when you get the time, for now adaptive meshing not needed
def mesh_from_pos(basename, msh_path, pos_file, lc_min, lc_max, step):
        
    gmsh.initialize()
    gmsh.open(msh_path)

    # merge pos into gmsh (creates a PostView)
    gmsh.merge(pos_file)

    # get the list of all current views
    views = gmsh.view.getTags()
    if not views:
        raise RuntimeError("No PostView found after merging POS file!")
    view_tag = views[-1]
    print(f"Using PostView {view_tag} from {pos_file} to drive adaptive remeshing")

    # Clear any existing fields safely
    try:
        gmsh.model.mesh.field.removeAll()
    except Exception:
        try:
            for ftag in gmsh.model.mesh.field.getTags():
                gmsh.model.mesh.field.remove(ftag)
        except Exception as e:
            print(f"⚠️ Could not clear fields cleanly: {e}")

    # Create a MathEval field that maps normalized indicator [0,1] → size [lc_min, lc_max]
    size_expr = f"{lc_max} - ({lc_max} - {lc_min}) * View[{view_tag}]"

    field_tag = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_tag, "F", size_expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    # Optional tuning for refinement
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    print(f"✅ Created adaptive background field (tag={field_tag}) with size expr: {size_expr}")

    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print("Gmsh meshing failed:", e)
        gmsh.finalize()
        raise

    next_msh = f"{basename}_{step+1}.msh"
    gmsh.write(next_msh)
    try:
        gmsh.write(next_msh.replace(".msh", ".vtk"))
    except Exception:
        pass
    

    gmsh.finalize()
    print(f"✅ Wrote refined mesh: {next_msh}")


def compute_mesh_sizes_from_geometry(gmsh_model, scale_factor=1/61.237, refine_ratio=4): #values such that for first geometry of hollow brick lc_min=0.5, lc_max=2.0
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_model.getBoundingBox(-1, -1)
    diag = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
    lc_max = diag * scale_factor
    lc_min = lc_max / refine_ratio
    return lc_min, lc_max

def grad_field_gen_new_bc(input_file, output_file="mesh.msh", n_adapt=2, E=2e11, nu=0.3):
    """
    Physics-driven adaptive meshing pipeline (corrected + field visualization).
    Signature left unchanged per request.

    Steps:
      - import STEP with OCC (gmsh)
      - generate initial mesh, write initial .msh/.vtk
      - for n_adapt iterations:
          -> load mesh in dolfinx, solve linear elasticity
          -> compute grad(|stress|) indicator per cell (DG0)
          -> normalize indicator, compute target size per cell
          -> write a .pos file with SP(centroid){indicator}
          -> import .pos into gmsh as a View -> PostView field
          -> create MathEval (size = lc_max - (lc_max-lc_min)*F{postview}) and set as background
          -> visualize & save a png of the target-size field (and attempt to show)
          -> remesh with gmsh, save next iteration mesh (.msh + .vtk)
      - write final outputs and return path to final .vtk
    """
    import os
    import numpy as np
    import math
    from pathlib import Path

    # imports that may be heavy / environment-dependent
    import gmsh
    from mpi4py import MPI
    from dolfinx import fem, mesh, io
    from dolfinx.fem import petsc
    import ufl

    # optional plotting
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except Exception:
        MATPLOTLIB_AVAILABLE = False

    comm = MPI.COMM_SELF
    basename = os.path.splitext(output_file)[0]

    # ---------- Initial mesh generation (with OCC import for STEP) ----------
    gmsh.initialize()
    gmsh.model.add("adaptive_part")

    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".step":
        print("Importing STEP as OCC shapes...")
        gmsh.model.occ.importShapes(input_file)
    

        gmsh.model.occ.synchronize()

        # Ensure a 3D volume exists; create one from surfaces if needed
        vols = gmsh.model.getEntities(dim=3)
        if not vols:
            surfs = gmsh.model.getEntities(dim=2)
            if not surfs:
                gmsh.finalize()
                raise RuntimeError("No surfaces found in geometry.")
            

            shell = gmsh.model.occ.addShell([s[1] for s in surfs])
            vol = gmsh.model.occ.addVolume([shell])
            gmsh.model.occ.synchronize()
            print(f"Created volume from surfaces (tag={vol}).")
        else:
            print("Found existing 3D volume(s).")

        # Add physical groups useful for gmshio
        vols = gmsh.model.getEntities(dim=3)
        surfs = gmsh.model.getEntities(dim=2)
        if vols:
            gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1)
            gmsh.model.setPhysicalName(3, 1, "Domain")
        if surfs:
            gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=2)
            gmsh.model.setPhysicalName(2, 2, "Boundary")

        lc_min, lc_max = compute_mesh_sizes_from_geometry(gmsh.model)
        # initial mesh generation & write
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
        print("Generating initial 3D mesh...")
        gmsh.model.mesh.generate(3)
        initial_msh = f"{basename}_0.msh"
        gmsh.write(initial_msh)
        # also write VTK for quick inspection
        try:
            gmsh.write(initial_msh.replace(".msh", ".vtk"))
        except Exception:
            pass
        gmsh.finalize()

        print(f"✅ Initial mesh written: {initial_msh}")
        print(f"Processed {input_file}")
    else:
        print(f"Skipped {input_file}, could not mesh it")
        
        return 0

    # ---------- adaptation loop ----------
    for step in range(n_adapt):
        print(f"\n=== Adaptive iteration {step} ===")
        msh_path = f"{basename}_{step}.msh" if step > 0 else initial_msh

        # Read mesh into dolfinx
        mesh_tuple = io.gmshio.read_from_msh(msh_path, comm, rank=0)
        msh, cell_tags, facet_tags = mesh_tuple

        # Ensure connectivity mapping from cells → their boundary faces, for BC application
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)

        print(f"Loaded mesh: nodes={msh.geometry.x.shape[0]}, cells={msh.topology.index_map(msh.topology.dim).size_local}")

        # --- elasticity solve (linear) ---
        # Elastic constants
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        def eps(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return 2 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(3)

        # Function space, trial and test functions
        V = fem.VectorFunctionSpace(msh, ("CG", 1))
        u = fem.Function(V)
        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)

        # Bilinear form (LHS)
        a = ufl.inner(sigma(du), eps(v)) * ufl.dx

        # Robust BC application
        # Compute geometry information
        geom = msh.geometry.x
        y_coords = geom[:, 1]
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        tol = 0.02 * (y_max - y_min)   # 2% of height range
        cos_thresh = 0.999               # allow curved surface normals max deviation 2.5 degrees 
        facet_normals = compute_normals(msh)
        
        def detect_top_bottom_facets(msh, facet_normals, y_min, y_max, cos_thresh=0.3, tol=1e-3):
            fdim = msh.topology.dim - 1
            msh.topology.create_connectivity(fdim, 0)
            conn = msh.topology.connectivity(fdim, 0)
            coords = msh.geometry.x

            # --- Only consider boundary facets ---
            boundary_facets = mesh.exterior_facet_indices(msh.topology)
            num_facets_total = msh.topology.index_map(fdim).size_local

            top_facets, bottom_facets = [], []
            for f in boundary_facets:
                verts = conn.links(f)
                y_verts = coords[verts, 1]
                n_y = np.abs(facet_normals[f, 1])

                # Require all vertices to lie near the top/bottom plane
                if np.allclose(y_verts, y_max, atol=tol) and n_y > cos_thresh:
                    top_facets.append(f)
                elif np.allclose(y_verts, y_min, atol=tol) and n_y > cos_thresh:
                    bottom_facets.append(f)

            # --- Diagnostics ---
            print(f"[detect_top_bottom_facets] Boundary facets: {len(boundary_facets)}/{num_facets_total}")
            print(f"Top facets: {len(top_facets)}, Bottom facets: {len(bottom_facets)}")

            return np.array(top_facets, dtype=np.int32), np.array(bottom_facets, dtype=np.int32)
        
        top_facets, bottom_facets = [], []
        iter = 0
        while(len(top_facets)<10 and iter <10):
            top_facets, bottom_facets = detect_top_bottom_facets(msh, facet_normals, y_min, y_max, cos_thresh, tol)
            cos_thresh -= 0.01
            iter += 1

        if len(top_facets) == 0: return 0
        # top_facets = mesh.locate_entities_boundary(
        #     msh, msh.topology.dim - 1,
        #     lambda x: is_top(y_max, x, facet_normals, cos_thresh, tol)
        # )
        # bottom_facets = mesh.locate_entities_boundary(
        #     msh, msh.topology.dim - 1,
        #     lambda x: is_bottom(y_min, x, facet_normals, cos_thresh, tol)
        # )

        #print(f"Detected {len(top_facets)} top facets, {len(bottom_facets)} bottom facets")

        run_bc_diagnostics(msh, facet_normals, top_facets, bottom_facets, y_min, y_max)
        #print(f'top facets: {top_facets}')
        #print(f'bottom facets: {bottom_facets}')
        # --- Tag facets ---

        facet_values = np.full(len(top_facets), 1, dtype=np.int32)
        facet_tags = mesh.meshtags(msh, msh.topology.dim - 1, top_facets, facet_values)

        # --- Define measure over tagged boundary ---
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

        # --- Apply Dirichlet BC on bottom (tag 2) ---
        bcdofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bottom_facets)
        bc = fem.dirichletbc(np.zeros(3, dtype=np.float64), bcdofs, V)

        # --- Apply Neumann load on top (tag 1) ---
        traction = fem.Constant(msh, np.array([0.0, -1e6, 0.0], dtype=np.float64))

        # --- Variational forms ---
        a = ufl.inner(sigma(du), eps(v)) * ufl.dx   # du = TrialFunction(V)
        L = ufl.inner(traction, v) * ds(1)

        # --- Solve using PETSc LinearProblem ---
        problem = petsc.LinearProblem(
            a, L, bcs=[bc])
        #     petsc_options={"ksp_type": "cg", "pc_type": "ilu"}
        # )
        uh = problem.solve()

        # --- compute stress scalar and grad magnitude ---
        stress_expr = sigma(uh)
        stress_scalar_expr = ufl.sqrt(ufl.inner(stress_expr, stress_expr))
        
        # project stress magnitude into scalar function space
        V0 = fem.FunctionSpace(msh, ("CG", 1))
        v0 = ufl.TestFunction(V0)
        u0 = ufl.TrialFunction(V0)

        a_proj = ufl.inner(u0, v0) * ufl.dx
        L_proj = ufl.inner(stress_scalar_expr, v0) * ufl.dx
        projector = fem.petsc.LinearProblem(a_proj, L_proj)
        stress_scalar = projector.solve()

        # now compute its gradient magnitude (symbolically, but on a projected field)
        grad_mag_expr = ufl.sqrt(ufl.inner(ufl.grad(stress_scalar), ufl.grad(stress_scalar)))

        # === DIAGNOSTIC A: check PDE solution and BCs/load ===
        print("---- DIAG A: solution & BCs ----")
        # uh vector stats
        try:
            ua = uh.x.array
            print("uh: n=", ua.size, "min=", ua.min(), "max=", ua.max(), "L2(norm)~", np.linalg.norm(ua))
        except Exception as e:
            print("Could not read uh array:", e)

        # check bottom facets and bcdofs
        print("bottom_facets count:", len(bottom_facets) if 'bottom_facets' in locals() else "MISSING")
        try:
            print("bcdofs length:", len(bcdofs))
        except Exception:
            print("bcdofs not available")

        # check top facets and traction
        print("top_facets count:", len(top_facets) if 'top_facets' in locals() else "MISSING")
        try:
            print("traction value:", np.array(traction.array) if hasattr(traction, "array") else traction)
        except Exception:
            try:
                print("traction (as Constant):", traction)
            except:
                pass

        # quick assembled integrals to see if fields are nonzero
        try:
            from dolfinx.fem import assemble_scalar

            # 🔍 DEBUG: check what ufl and assemble_scalar really are
            import inspect, ufl
            print("DEBUG --- checking ufl environment ---")
            print("ufl module path:", getattr(ufl, "__file__", "no file attr"))
            print("ufl.form type:", type(ufl.form))
            print("assemble_scalar type:", type(assemble_scalar))
            print("locals with 'ufl':", [k for k in locals() if k == "ufl"])
            print("globals with 'ufl':", [k for k in globals() if k == "ufl"])
            print("-------------------------------")
            
            val_s = assemble_scalar(fem.form(stress_scalar * ufl.dx))
            val_g = assemble_scalar(fem.form(grad_mag_expr * ufl.dx))
            print("Assembled stress_scalar integral:", val_s)
            print("Assembled grad_mag_expr integral:", val_g)
        except Exception as e:
            print("Assemble diagnostics failed:", e)
        print("---- end DIAG A ----\n")
        
        
        # project grad_mag_expr to DG0 (cell-wise)
        V0 = fem.FunctionSpace(msh, ("DG", 0))
        a0 = ufl.inner(ufl.TrialFunction(V0), ufl.TestFunction(V0)) * ufl.dx
        L0 = ufl.inner(grad_mag_expr, ufl.TestFunction(V0)) * ufl.dx
        from dolfinx.fem.petsc import LinearProblem as LP
        cell_indicator = LP(a0, L0).solve()
        cell_vals = np.array(cell_indicator.x.array, copy=True)  # length = num_cells


        # Save stress tensor (optional) for inspection
        try:
            with io.XDMFFile(comm, f"stress_{step}.xdmf", "w") as xdmf:
                xdmf.write_mesh(msh)
                # create a tensor function for stress by projecting the expression
                Tensor = ufl.TensorElement("DG", msh.ufl_cell(), 0)
                Wt = fem.FunctionSpace(msh, Tensor)
                a_proj = ufl.inner(ufl.TrialFunction(Wt), ufl.TestFunction(Wt)) * ufl.dx
                L_proj = ufl.inner(stress_expr, ufl.TestFunction(Wt)) * ufl.dx
                stress_proj = LP(a_proj, L_proj).solve()
                stress_proj.name = "stress_tensor"
                xdmf.write_function(stress_proj)
        except Exception as e:
            print("Warning: Could not write stress XDMF -", e)

        # ---------- Build .pos from cell centroids & cell indicator ----------
        # get cell->node connectivity (attempt robust extraction)
        try:
            if not msh.topology.has_connectivity(msh.topology.dim, 0):
                msh.topology.create_connectivity(msh.topology.dim, 0)
            conn = msh.topology.connectivity(msh.topology.dim, 0)
            conn_arr = np.array(conn.array, dtype=np.int64)
            num_cells = int(msh.topology.index_map(msh.topology.dim).size_local)
            if num_cells == 0:
                raise RuntimeError("No cells found in mesh.")
            nodes_per_cell = int(conn_arr.size // num_cells)
            cells_to_nodes = conn_arr.reshape((num_cells, nodes_per_cell))
        except Exception:
            # fallback: try iterating cells via topology (slower)
            print("Warning: connectivity.array not available; using slower fallback.")
            num_cells = int(msh.topology.index_map(msh.topology.dim).size_local)
            cells_to_nodes = []
            for ci in range(num_cells):
                try:
                    node_list = msh.topology.connectivity(msh.topology.dim, 0).links(ci)
                except Exception:
                    # last-resort: attempt using cells from topology (may fail on some versions)
                    node_list = []
                cells_to_nodes.append(np.array(node_list, dtype=np.int64))
            # convert to ragged list; handle below

        # compute centroids
        geox = np.array(msh.geometry.x)
        centroids = []
        for ci, nodes in enumerate(cells_to_nodes):
            nodes = np.asarray(nodes, dtype=int)
            if nodes.size == 0:
                # skip; should not happen
                centroids.append(np.array([0.0, 0.0, 0.0]))
            else:
                centroids.append(geox[nodes].mean(axis=0))
        centroids = np.array(centroids)


        # === DIAGNOSTIC B: check cell indicator that will be written to .pos ===
        print("---- DIAG B: cell indicator ----")
        print("cell_vals length:", cell_vals.size)
        if cell_vals.size > 0:
            print("cell_vals min/max/mean:", float(np.min(cell_vals)), float(np.max(cell_vals)), float(np.mean(cell_vals)))
            print("cell_vals sample[0:10]:", cell_vals[:10])
        else:
            print("cell_vals is empty!")

        # check centroid count vs cell_vals
        print("num centroids:", centroids.shape[0])
        if centroids.shape[0] != cell_vals.size:
            print("WARNING: centroids count != cell_vals count!", centroids.shape[0], cell_vals.size)
        else:
            print("centroids and cell_vals lengths match")

        # detailed floating precision print for min/max
        try:
            vmin = float(np.min(cell_vals))
            vmax = float(np.max(cell_vals))
            print("vmin/vmax (high precision):", format(vmin, ".12e"), format(vmax, ".12e"))
        except Exception:
            pass
        print("---- end DIAG B ----\n")

        # normalize cell indicator to 0..1
        vals = cell_vals
        if vals.size == 0:
            raise RuntimeError("Cell indicator is empty.")
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax - vmin < 1e-14:
            cell_norm = np.zeros_like(vals, dtype=float)
        else:
            cell_norm = (vals - vmin) / (vmax - vmin)

        # compute target sizes (size small where indicator large)
        target_sizes = lc_max - (lc_max - lc_min) * cell_norm  # size per cell

        # write pos file using cell centroid positions and normalized indicator (0..1)
        pos_file = f"{basename}_grad_{step}.pos"
        with open(pos_file, "w") as fpos:
            fpos.write('View "grad" {\n')
            for ci, c in enumerate(centroids):
                val = float(cell_norm[ci])  # normalized indicator in [0,1]
                fpos.write(f"SP({c[0]},{c[1]},{c[2]}){{{val}}};\n")
            fpos.write("};\n")
        print(f"Wrote pos file with {len(centroids)} points: {pos_file}")
    
    return 1
    # with open('processed_step_files.txt', 'a') as f:
    #     f.write(f"{input_file}\n")

TIME_LIMIT = 300  # seconds (5 minutes)

def run_grad_field(cad_file_path):
    
    return grad_field_gen_new_bc(
        cad_file_path,
        output_file=f"{cad_file_path.split('.')[0]}.msh",
        n_adapt=1
    )

if __name__ == "__main__":

    #/data/1bali/Other_LLM_projects/multi_view_3DQA/2D_FE_Mesh_2.jpg
    path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-geometries"
    CAD_mesh_path_dir = "/data/1bali/Other_LLM_projects/multi_view_3DQA/FreeCAD-library-meshes"
    processed_files_, timeout_files_ = [], []
    for CAD_Folder in tqdm(os.listdir(path_dir)):
        ## Extract CAD file and convert to STEP if needed
        cad_file_valid = 1
        for cad_file in os.listdir(f'{path_dir}/{CAD_Folder}'):
            if cad_file.lower().endswith('.fcstd'):
                try:
                    fcstd_to_step(f"{path_dir}/{CAD_Folder}/{cad_file}", f"{path_dir}/{CAD_Folder}/{cad_file.split('.')[0]}.step")
                    cad_file = f"{cad_file.split('.')[0]}.step"
                except:
                    cad_file_valid = 0
                break
            elif cad_file.lower().endswith('.step'): break
        
        if not cad_file.lower().endswith('.step'): 
            print(f'{cad_file} STEP file not found, continuing')
            continue
            
        cad_file_path = f"{path_dir}/{CAD_Folder}/{cad_file}"
        
        if not cad_file_valid:
            with open('invalid_cad.txt', 'a') as f:
                f.write(f'{cad_file_path}\n')

        processed_files = []
        if os.path.exists('processed_step_files.txt'):
            with open('processed_step_files.txt', 'r') as f:
                processed_files = f.read().splitlines()
        else:
            with open('processed_step_files.txt', 'w') as f:
                f.write('')

        with open('skipped_due_to_timeout.txt', 'r') as f:
            skipped_files = f.read().splitlines()

        if cad_file_path in processed_files or cad_file_path in skipped_files:
            print(f"Skipping already processed file: {cad_file_path}")
            continue  # Skip to the next file
        
        p = mp.Process(target=run_grad_field, args=(cad_file_path,))
        p.start()
        p.join(TIME_LIMIT)

        
        if p.is_alive():
            print(f"⏰ Timeout: {cad_file_path} exceeded {TIME_LIMIT}s, skipping...")
            p.terminate()
            p.join()
            with open("skipped_due_to_timeout.txt", "a") as f:
                f.write(f"{cad_file_path}\n")
            timeout_files_.append(cad_file_path)
        else:
            with open("processed_step_files.txt", "a") as f:
                f.write(f"{cad_file_path}\n")
        
        ## Render views from the simulated pos file
        pos_file_path = f'{path_dir}/{CAD_Folder}/{CAD_Folder}_grad_0.pos'
        if not os.path.exists(pos_file_path):
            print(f"pos file does not exist: {pos_file_path}")
            continue
        output_dir = f'{path_dir}/{CAD_Folder}/renders_pyvista'

        # render_plotly_views(
        #     pos_file_path,
        #     output_dir=output_dir,
        #     n_azimuth=12,
        #     n_elevation=1,
        #     zoom=1.6)
        try:
            render_pyvista_views(
                pos_file_path,
                output_dir=output_dir,
                n_azimuth=12,
                n_elevation=5,
                zoom=4.5)
            with open('valid_pos_files.txt', 'a') as f:
                f.write(f'{pos_file_path}\n')
        except:
            print(f'Could not process {pos_file_path}')
            with open('invalid_pos_files.txt', 'a') as f:
                f.write(f'{pos_file_path}\n')

         ## transfer the mesh file also to the folder
        import shutil
        
        shutil.copy(f'{CAD_mesh_path_dir}/{CAD_Folder}.obj', f'{path_dir}/{CAD_Folder}/renders_pyvista/{CAD_Folder}.obj')


        # grad_field_gen_new_bc(cad_file_path,
        # output_file=f"{cad_file_path.split('.')[0]}.msh",
        # n_adapt=1)

        # grad_field_gen(cad_file_path, output_file=f"{cad_file_path.split('.')[0]}.msh",
        #                 n_adapt=1)
        # lc_min=0.5,
        # lc_max=2.0,
