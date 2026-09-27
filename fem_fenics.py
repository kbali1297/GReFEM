#!/usr/bin/env python3
"""
fenics_multiload_compare.py

Extends fenics_compression_compare.py with two additional load cases:

  run_fenics_bending(input_msh, ...)
      Cantilever bending: bottom face fixed (all DOFs = 0),
      tip face gets a prescribed transverse (y) displacement.
      Stress concentrates at root fillets and top/bottom flanges.

  run_fenics_torsion(input_msh, ...)
      Torsion: one end face fixed (all DOFs = 0),
      opposite end face gets a rigid-body rotation about the beam axis (z).
      Stress concentrates at outer surface, especially near geometric discontinuities.

  run_fenics_compression(input_msh, ...)   [unchanged from original]

  compute_disp_error(candidate_mshs, ref_msh, load_case="compression"|"bending"|"torsion", ...)
      Dispatches to the correct solver and then computes L2/energy errors
      against the reference solution.

All three solvers share the same:
  - Linear elasticity weak form (sigma/eps)
  - Output format (_sol.xdmf, _stress.xdmf, _strain.xdmf)
  - Error computation pipeline (ParaView interpolation + FEniCS assembly)

Usage:
    python fenics_multiload_compare.py --load_case bending --meshA ... --meshRef ...
"""

import os
import meshio
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from dolfinx import fem, io
from dolfinx.fem import petsc
import h5py
import ufl
from petsc4py import PETSc
PETSc.Options().clear()
import basix
import gmsh
import xml.etree.ElementTree as ET
import subprocess

# ============================================================
# Shared constitutive utilities
# ============================================================

def eps(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, nu=0.3, E=2e11):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(3)


def compute_mesh_sizes_from_geometry(gmsh_model, scale_factor=1/120, refine_ratio=4):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_model.getBoundingBox(-1, -1)
    diag = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
    lc_max = diag * scale_factor
    lc_min = lc_max / refine_ratio
    print(f'zz stress Coarse Level Current: {lc_max}')
    return lc_min, lc_max


# ============================================================
# ZZ stress indicator
# ============================================================

def return_zz_field(input_tet3D_mesh, uh, output_file=None):
    # --- Handle loading from file if uh is a path string ---
    if isinstance(uh, str):
        u_xdmf_path = uh
        with io.XDMFFile(MPI.COMM_SELF, u_xdmf_path, "r") as xdmf:
            msh = xdmf.read_mesh(name="mesh")
        gdim = msh.geometry.dim
        # Define Space: family, degree, shape
        V = fem.functionspace(msh, ("CG", 1, (gdim,)))
        uh = fem.Function(V)
        h5_path = u_xdmf_path.replace(".xdmf", ".h5")
        with h5py.File(h5_path, "r") as h5:
            uh.x.array[:] = h5["/Function/u/0"][:].reshape(-1)
    else:
        msh = uh.function_space.mesh
        gdim = msh.geometry.dim

    stress_expr = sigma(uh)
    dirname = os.path.dirname(input_tet3D_mesh)
    basename = os.path.basename(input_tet3D_mesh).replace('.msh', '')

    # --- 1. Project stress to DG0 (Discontinuous Galerkin at centroids) ---
    # NEW API: Define space directly with (family, degree, (shape))
    W_DG0 = fem.functionspace(msh, ("DG", 0, (gdim, gdim)))
    vT = ufl.TestFunction(W_DG0)
    uT = ufl.TrialFunction(W_DG0)
    
    # Solve projection: σh is the discontinuous stress field
    prob1 = fem.petsc.LinearProblem(ufl.inner(uT, vT) * ufl.dx, ufl.inner(stress_expr, vT) * ufl.dx)
    σh = prob1.solve()

    # --- 2. Smooth stress to CG1 (Continuous Galerkin at nodes) ---
    # This is the "Recovery" step of ZZ
    W_CG1 = fem.functionspace(msh, ("CG", 1, (gdim, gdim)))
    vS = ufl.TestFunction(W_CG1)
    uS = ufl.TrialFunction(W_CG1)
    
    prob2 = fem.petsc.LinearProblem(ufl.inner(uS, vS) * ufl.dx, ufl.inner(σh, vS) * ufl.dx)
    σstar = prob2.solve()

    # --- 3. Compute error indicator ||σ* - σh|| ---
    diff_expr = ufl.sqrt(ufl.inner(σstar - σh, σstar - σh))
    W0 = fem.functionspace(msh, ("DG", 0))
    v0 = ufl.TestFunction(W0)
    u0 = ufl.TrialFunction(W0)
    
    prob3 = fem.petsc.LinearProblem(ufl.inner(u0, v0) * ufl.dx, ufl.inner(diff_expr, v0) * ufl.dx)
    zz_stress = prob3.solve()

    # --- 4. Extract data for .pos file ---
    cell_vals_zz_stress_mag = np.array(zz_stress.x.array, copy=True)
    cells_to_nodes = return_cells_to_nodes(msh)
    geox = np.array(msh.geometry.x)
    
    # Calculate centroids for the .pos markers
    centroids = np.array([geox[np.asarray(nodes, dtype=int)].mean(axis=0)
                          for nodes in cells_to_nodes])

    pos_file = output_file or f"{dirname}/{basename}_zz.pos"
    write_pos_file(pos_file, centroids, cell_vals_zz_stress_mag)
    
    return pos_file


def write_pos_file(pos_file, centroids, cell_vals):
    vals = cell_vals
    if vals.size == 0:
        raise RuntimeError("Cell indicator is empty.")
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    cell_norm = np.zeros_like(vals, dtype=float) if vmax - vmin < 1e-14 \
        else (vals - vmin) / (vmax - vmin)
    with open(pos_file, "w") as fpos:
        fpos.write('View "grad" {\n')
        for ci, c in enumerate(centroids):
            fpos.write(f"SP({c[0]},{c[1]},{c[2]}){{{float(cell_norm[ci])}}};\n")
        fpos.write("};\n")
    print(f"Wrote pos file with {len(centroids)} points: {pos_file}")


def return_cells_to_nodes(msh):
    try:
        if not msh.topology.has_connectivity(msh.topology.dim, 0):
            msh.topology.create_connectivity(msh.topology.dim, 0)
        conn = msh.topology.connectivity(msh.topology.dim, 0)
        conn_arr = np.array(conn.array, dtype=np.int64)
        num_cells = int(msh.topology.index_map(msh.topology.dim).size_local)
        if num_cells == 0:
            raise RuntimeError("No cells found in mesh.")
        nodes_per_cell = int(conn_arr.size // num_cells)
        return conn_arr.reshape((num_cells, nodes_per_cell))
    except Exception:
        print("Warning: connectivity.array not available; using slower fallback.")
        num_cells = int(msh.topology.index_map(msh.topology.dim).size_local)
        cells_to_nodes = []
        for ci in range(num_cells):
            try:
                node_list = msh.topology.connectivity(msh.topology.dim, 0).links(ci)
            except Exception:
                node_list = []
            cells_to_nodes.append(np.array(node_list, dtype=np.int64))
        return np.array(cells_to_nodes, dtype=np.int64)


# ============================================================
# Geometry / boundary helpers
# ============================================================

def compute_normals(msh):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    facet_conn = msh.topology.connectivity(fdim, 0)
    num_facets = msh.topology.index_map(fdim).size_local
    offsets = facet_conn.offsets
    indices = facet_conn.array
    mask = np.diff(offsets) >= 3
    v0 = indices[offsets[:-1][mask]]
    v1 = indices[offsets[:-1][mask] + 1]
    v2 = indices[offsets[:-1][mask] + 2]
    coords = msh.geometry.x
    p0, p1, p2 = coords[v0], coords[v1], coords[v2]
    n = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-12
    n[valid] /= norms[valid, np.newaxis]
    normals = np.zeros((num_facets, 3))
    normals[mask] = n
    return normals


def detect_top_bottom_facets(msh, facet_normals, y_min, y_max,
                              cos_thresh=0.3, tol=1e-3):
    """Detect facets whose outward normal is ±y (used for compression/bending)."""
    from dolfinx import mesh as dmesh_mod
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x
    boundary_facets = dmesh_mod.exterior_facet_indices(msh.topology)
    top_facets, bottom_facets = [], []
    for f in boundary_facets:
        verts = conn.links(f)
        y_verts = coords[verts, 1]
        n_y = np.abs(facet_normals[f, 1])
        if n_y >= cos_thresh:
            if np.all(np.abs(y_verts - y_max) < tol):
                top_facets.append(f)
            elif np.all(np.abs(y_verts - y_min) < tol):
                bottom_facets.append(f)
    num_facets_total = msh.topology.index_map(fdim).size_local
    print(f"[detect_top_bottom_facets] Boundary: {len(boundary_facets)}/{num_facets_total} "
          f"| top: {len(top_facets)} | bottom: {len(bottom_facets)}")
    return np.array(top_facets, dtype=np.int32), np.array(bottom_facets, dtype=np.int32)


def detect_end_facets_z(msh, facet_normals, cos_thresh=0.85, tol_frac=0.02):
    """
    Detect facets whose outward normal is ±z (beam axis).
    Returns fixed_facets (z_min) and loaded_facets (z_max).
    """
    from dolfinx import mesh as dmesh_mod
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x

    z_coords = coords[:, 2]
    z_min, z_max = float(z_coords.min()), float(z_coords.max())
    tol = tol_frac * (z_max - z_min)

    boundary_facets = dmesh_mod.exterior_facet_indices(msh.topology)
    fixed_facets, loaded_facets = [], []

    for f in boundary_facets:
        verts = conn.links(f)
        z_verts = coords[verts, 2]
        n_z = np.abs(facet_normals[f, 2])
        if n_z >= cos_thresh:
            if np.all(np.abs(z_verts - z_min) < tol):
                fixed_facets.append(f)
            elif np.all(np.abs(z_verts - z_max) < tol):
                loaded_facets.append(f)

    print(f"[detect_end_facets_z] fixed(z_min): {len(fixed_facets)} "
          f"| loaded(z_max): {len(loaded_facets)}")
    return np.array(fixed_facets, dtype=np.int32), np.array(loaded_facets, dtype=np.int32)


# ============================================================
# Shared solver boilerplate
# ============================================================

def _setup_mesh(input_msh):
    filesave_path = input_msh.split(".msh")[0]
    m = meshio.read(input_msh)
    meshio.write(input_msh, m, file_format="gmsh22")
    msh, _, _ = io.gmshio.read_from_msh(input_msh, MPI.COMM_SELF, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    # facet->cell connectivity: required by exterior_facet_indices; volume-only
    # meshes (e.g. AMBER outputs without boundary triangles) don't get it
    # implicitly during read_from_msh.
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    print(f"Mesh: {msh.geometry.x.shape[0]} nodes, "
          f"{msh.topology.index_map(msh.topology.dim).size_local} cells")
    return msh, filesave_path


def _petsc_opts():
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1e-8
    opts["pc_type"] = "gamg"
    opts["pc_gamg_type"] = "agg"
    opts["pc_gamg_agg_nsmooths"] = 1
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"


def _solve_and_write(msh, V, bcs, filesave_path):
    du = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)

    a = ufl.inner(sigma(du), eps(v)) * ufl.dx
    zero_vec = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=np.float64))
    L = ufl.inner(zero_vec, v) * ufl.dx

    _petsc_opts()
    problem = petsc.LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()

    dim = msh.topology.dim
    Wt = fem.functionspace(msh, ("DG", 0, (dim, dim)))

    def project_tensor(expr):
        a_p = ufl.inner(ufl.TrialFunction(Wt), ufl.TestFunction(Wt)) * ufl.dx
        L_p = ufl.inner(expr, ufl.TestFunction(Wt)) * ufl.dx
        return petsc.LinearProblem(a_p, L_p).solve()

    stress_tensor = project_tensor(sigma(uh))
    strain_tensor = project_tensor(eps(uh))

    comm = MPI.COMM_SELF
    u_xdmf      = f'{filesave_path}_sol.xdmf'
    stress_xdmf = f'{filesave_path}_stress.xdmf'
    strain_xdmf = f'{filesave_path}_strain.xdmf'

    uh.name = "u"
    with io.XDMFFile(comm, u_xdmf, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    stress_tensor.name = "stress_tensor"
    with io.XDMFFile(comm, stress_xdmf, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(stress_tensor)

    strain_tensor.name = "strain_tensor"
    with io.XDMFFile(comm, strain_xdmf, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(strain_tensor)

    print(f"Wrote: {u_xdmf}, {stress_xdmf}, {strain_xdmf}")
    return u_xdmf, stress_xdmf, strain_xdmf, uh


# ============================================================
# LOAD CASE 1 — Compression
# ============================================================

def run_fenics_compression(input_msh, disp_frac=0.01, E=2e11, nu=0.3):
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)
    top_facets, bottom_facets = [], []
    cos_thresh = 0.9
    for _ in range(10):
        top_facets, bottom_facets = detect_top_bottom_facets(
            msh, facet_normals, y_min, y_max, cos_thresh, tol)
        if len(top_facets) >= 10:
            break
        cos_thresh -= 0.01

    if len(bottom_facets) == 0:
        print('Could not detect bottom facets. Aborting.')
        return None, None, None, None, None

    bc_bottom = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, bottom_facets), V)

    disp_val = -disp_frac * (y_max - y_min)
    V_y = V.sub(1)
    bc_top = fem.dirichletbc(
        np.array(disp_val, dtype=np.float64),
        fem.locate_dofs_topological(V_y, msh.topology.dim - 1, top_facets), V_y)

    bcs = [bc_bottom, bc_top]
    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh


# ============================================================
# LOAD CASE 2 — Bending (cantilever)
# ============================================================

def run_fenics_bending(input_msh, disp_frac=0.01, bend_axis=0, E=2e11, nu=0.3):
    """
    Pure Bending applied to the top face.

    Boundary conditions
    -------------------
    fixed end  (y_min face) : all DOFs = 0
    loaded end (y_max face) : u_y rotates around the face's centroid.
                              Nodes to the left go down, nodes to the right go up.
                              u_x and u_z are left completely FREE so the beam can
                              naturally curve and translate.
    """
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)

    # --- detect clamped (y_min) and tip (y_max) faces ---
    top_facets, bottom_facets = [], []
    cos_thresh = 0.9
    for _ in range(10):
        top_facets, bottom_facets = detect_top_bottom_facets(
            msh, facet_normals, y_min, y_max, cos_thresh, tol)
        if len(bottom_facets) >= 3 and len(top_facets) >= 3:
            break
        cos_thresh -= 0.05

    if len(bottom_facets) == 0 or len(top_facets) == 0:
        print("Could not detect top/bottom facets for bending. Aborting.")
        return None, None, None, None, None

    # --- BC 1: Clamp bottom face (All DOFs = 0) ---
    bc_fixed = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, bottom_facets), V)

    # --- BC 2: Twist/Bending profile on top face (ONLY Y-direction) ---
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x

    # Find the centroid X of the top face
    loaded_nodes = set()
    for f in top_facets:
        for n in conn.links(f):
            loaded_nodes.add(int(n))
    cx = float(coords[list(loaded_nodes), 0].mean())

    # To apply a spatially varying BC to ONLY the Y-component in FEniCSx, 
    # we extract the Y-subspace and collapse it into a scalar space.
    V_y = V.sub(1)
    Q, Q_map = V_y.collapse()
    u_bend = fem.Function(Q)

    # Define the rotational profile: (X - Center) * angle
    # disp_frac acts as our rotation angle in radians
    def bend_profile(x):
        return (x[0] - cx) * disp_frac

    u_bend.interpolate(bend_profile)

    # Apply this ONLY to the Y DOFs of the top face
    dof_indices = fem.locate_dofs_topological((V_y, Q), fdim, top_facets)
    bc_tip = fem.dirichletbc(u_bend, dof_indices, V_y)

    bcs = [bc_fixed, bc_tip]
    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh


# ============================================================
# LOAD CASE 3 — Torsion
# ============================================================

def run_fenics_torsion(input_msh, twist_angle_deg=1.0, E=2e11, nu=0.3):
    """
    Pure torsion about the Y-axis.

    Boundary conditions
    -------------------
    fixed end  (y_min face) : all DOFs = 0
    loaded end (y_max face) : u_x and u_z rotate around the face's centroid.
                              u_y is left completely FREE so the cross-section
                              can naturally warp out-of-plane (St. Venant torsion).
    """
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)
    
    top_facets, bottom_facets = [], []
    cos_thresh = 0.9
    for _ in range(10):
        top_facets, bottom_facets = detect_top_bottom_facets(
            msh, facet_normals, y_min, y_max, cos_thresh, tol)
        if len(bottom_facets) >= 3 and len(top_facets) >= 3:
            break
        cos_thresh -= 0.05

    if len(bottom_facets) == 0 or len(top_facets) == 0:
        print("Could not detect top/bottom facets for torsion. Aborting.")
        return None, None, None, None, None

    # --- BC 1: Clamp bottom face (All DOFs = 0) ---
    bc_fixed = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, bottom_facets), V)

    # --- BC 2: Twist profile on top face (ONLY X and Z directions) ---
    theta = np.deg2rad(twist_angle_deg)
    
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x

    # Find the centroid (X, Z) of the top face
    loaded_nodes = set()
    for f in top_facets:
        for n in conn.links(f):
            loaded_nodes.add(int(n))
    loaded_nodes = list(loaded_nodes)
    cx = float(coords[loaded_nodes, 0].mean())
    cz = float(coords[loaded_nodes, 2].mean())

    # Extract X and Z subspaces
    V_x = V.sub(0)
    V_z = V.sub(2)
    Q_x, _ = V_x.collapse()
    Q_z, _ = V_z.collapse()
    u_twist_x = fem.Function(Q_x)
    u_twist_z = fem.Function(Q_z)

    # Define the rotational profile based on small angle approximation:
    # u_x = theta * (z - cz)
    # u_z = -theta * (x - cx)
    def twist_profile_x(x):
        return theta * (x[2] - cz)
        
    def twist_profile_z(x):
        return -theta * (x[0] - cx)

    u_twist_x.interpolate(twist_profile_x)
    u_twist_z.interpolate(twist_profile_z)

    # Apply BCs strictly to the X and Z DOFs, leaving Y free!
    dof_indices_x = fem.locate_dofs_topological((V_x, Q_x), fdim, top_facets)
    dof_indices_z = fem.locate_dofs_topological((V_z, Q_z), fdim, top_facets)
    
    bc_tip_x = fem.dirichletbc(u_twist_x, dof_indices_x, V_x)
    bc_tip_z = fem.dirichletbc(u_twist_z, dof_indices_z, V_z)

    bcs = [bc_fixed, bc_tip_x, bc_tip_z]
    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh

# ============================================================
# LOAD CASE 4 — Shear (Guided Cantilever)
# ============================================================

def run_fenics_shear(input_msh, disp_frac=0.01, E=2e11, nu=0.3):
    """
    Transverse Shear (Guided Cantilever) applied to the top face.

    Boundary conditions
    -------------------
    fixed end  (y_min face) : all DOFs = 0
    loaded end (y_max face) : u_x = +disp_frac * beam_length
                              u_y and u_z are left free.
                              
    Because u_x is forced to be a uniform constant across the entire face,
    the face is not allowed to naturally tilt/rotate in the X-direction.
    This creates a shear-dominated "S" curvature with high stress 
    concentrations at both the fixed root and the loaded tip.
    """
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)

    # --- detect clamped (y_min) and tip (y_max) faces ---
    top_facets, bottom_facets = [], []
    cos_thresh = 0.9
    for _ in range(10):
        top_facets, bottom_facets = detect_top_bottom_facets(
            msh, facet_normals, y_min, y_max, cos_thresh, tol)
        if len(bottom_facets) >= 3 and len(top_facets) >= 3:
            break
        cos_thresh -= 0.05

    if len(bottom_facets) == 0 or len(top_facets) == 0:
        print("Could not detect top/bottom facets for shear. Aborting.")
        return None, None, None, None, None

    # --- BC 1: Clamp bottom face (All DOFs = 0) ---
    bc_fixed = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, bottom_facets), V)

    # --- BC 2: Uniform translation in +X direction on top face ---
    beam_length = y_max - y_min
    tip_disp = abs(disp_frac) * beam_length

    # V.sub(0) corresponds to the X-component
    V_shear = V.sub(0)
    bc_tip = fem.dirichletbc(
        np.array(tip_disp, dtype=np.float64),
        fem.locate_dofs_topological(V_shear, msh.topology.dim - 1, top_facets),
        V_shear)

    bcs = [bc_fixed, bc_tip]
    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh

# ============================================================
# LOAD CASE 5 — Bending + Compression
# ============================================================

def run_fenics_bending_compression(input_msh, disp_frac=0.01, E=2e11, nu=0.3):
    """
    Combined Load: 
      - Fixed Support (-X face)
      - Compression (-X direction) on the (+X face)
      - Bending (Y displacement profile) on the (+Y face)
    """
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    x_min, x_max = float(geom[:, 0].min()), float(geom[:, 0].max())
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol_x = 0.02 * (x_max - x_min)
    tol_y = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)

    # 1. Detect required faces
    top_facets, left_facets, right_facets = [], [],[]
    
    # Find +Y (Top face)
    cos_thresh = 0.9
    for _ in range(10):
        tf, _ = detect_top_bottom_facets(msh, facet_normals, y_min, y_max, cos_thresh, tol_y)
        if len(tf) >= 3:
            top_facets = tf
            break
        cos_thresh -= 0.05

    # Find -X (Left) and +X (Right) faces
    cos_thresh = 0.9
    for _ in range(10):
        lf, rf = detect_left_right_facets(msh, facet_normals, x_min, x_max, cos_thresh, tol_x)
        if len(lf) >= 3 and len(rf) >= 3:
            left_facets = lf
            right_facets = rf
            break
        cos_thresh -= 0.05

    if len(top_facets) == 0 or len(left_facets) == 0 or len(right_facets) == 0:
        print("Could not detect necessary facets (+Y, -X, +X) for bending_compression. Aborting.")
        return None, None, None, None, None

    # --- BC 1: Clamp left face (-X face, all DOFs = 0) ---
    bc_fixed = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, left_facets), V)

    # --- BC 2: Compression on right face (+X face pushing in -X direction) ---
    V_x = V.sub(0)
    comp_disp = -disp_frac * (x_max - x_min)
    bc_comp = fem.dirichletbc(
        np.array(comp_disp, dtype=np.float64),
        fem.locate_dofs_topological(V_x, msh.topology.dim - 1, right_facets), V_x)

    # --- BC 3: Bending on top face (+Y face) ---
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    
    # Find centroid of top face to pivot the bending
    loaded_nodes = set()
    for f in top_facets:
        for n in conn.links(f):
            loaded_nodes.add(int(n))
    cx = float(geom[list(loaded_nodes), 0].mean())

    V_y = V.sub(1)
    Q_y, _ = V_y.collapse()
    u_bend = fem.Function(Q_y)

    def bend_profile(x):
        return (x[0] - cx) * disp_frac

    u_bend.interpolate(bend_profile)
    dof_indices_y = fem.locate_dofs_topological((V_y, Q_y), fdim, top_facets)
    bc_bend = fem.dirichletbc(u_bend, dof_indices_y, V_y)

    bcs = [bc_fixed, bc_comp, bc_bend]
    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh


# ============================================================
# LOAD CASE 6 — Torsion + Compression
# ============================================================

def run_fenics_torsion_compression(input_msh, twist_angle_deg=1.0, disp_frac=0.01, E=2e11, nu=0.3):
    """
    Combined Load: 
      - Fixed Support (-X face)
      - Compression (-X direction) on the (+X face)
      - Torsion (Twist around Y axis) on the (+Y face)
    """
    msh, filesave_path = _setup_mesh(input_msh)
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))

    geom = msh.geometry.x
    x_min, x_max = float(geom[:, 0].min()), float(geom[:, 0].max())
    y_min, y_max = float(geom[:, 1].min()), float(geom[:, 1].max())
    tol_x = 0.02 * (x_max - x_min)
    tol_y = 0.02 * (y_max - y_min)

    facet_normals = compute_normals(msh)

    # 1. Detect required faces
    top_facets, left_facets, right_facets = [], [],[]
    
    cos_thresh = 0.9
    for _ in range(10):
        tf, _ = detect_top_bottom_facets(msh, facet_normals, y_min, y_max, cos_thresh, tol_y)
        if len(tf) >= 3:
            top_facets = tf
            break
        cos_thresh -= 0.05

    cos_thresh = 0.9
    for _ in range(10):
        lf, rf = detect_left_right_facets(msh, facet_normals, x_min, x_max, cos_thresh, tol_x)
        if len(lf) >= 3 and len(rf) >= 3:
            left_facets = lf
            right_facets = rf
            break
        cos_thresh -= 0.05

    if len(top_facets) == 0 or len(left_facets) == 0 or len(right_facets) == 0:
        print("Could not detect necessary facets (+Y, -X, +X) for torsion_compression. Aborting.")
        return None, None, None, None, None

    # --- BC 1: Clamp left face (-X face) ---
    bc_fixed = fem.dirichletbc(
        np.zeros(3, dtype=np.float64),
        fem.locate_dofs_topological(V, msh.topology.dim - 1, left_facets), V)

    # --- BC 2: Compression on right face (+X face pushing in -X direction) ---
    V_x = V.sub(0)
    comp_disp = -disp_frac * (x_max - x_min)
    bc_comp = fem.dirichletbc(
        np.array(comp_disp, dtype=np.float64),
        fem.locate_dofs_topological(V_x, msh.topology.dim - 1, right_facets), V_x)

    # --- BC 3: Torsion on top face (+Y face) ---
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    
    loaded_nodes = set()
    for f in top_facets:
        for n in conn.links(f):
            loaded_nodes.add(int(n))
    loaded_nodes = list(loaded_nodes)
    
    cx = float(geom[loaded_nodes, 0].mean())
    cz = float(geom[loaded_nodes, 2].mean())

    theta = np.deg2rad(twist_angle_deg)
    
    V_z = V.sub(2)
    Q_x, _ = V_x.collapse()
    Q_z, _ = V_z.collapse()
    
    u_twist_x = fem.Function(Q_x)
    u_twist_z = fem.Function(Q_z)

    def twist_profile_x(x):
        return theta * (x[2] - cz)
        
    def twist_profile_z(x):
        return -theta * (x[0] - cx)

    u_twist_x.interpolate(twist_profile_x)
    u_twist_z.interpolate(twist_profile_z)

    dof_indices_x_top = fem.locate_dofs_topological((V_x, Q_x), fdim, top_facets)
    dof_indices_z_top = fem.locate_dofs_topological((V_z, Q_z), fdim, top_facets)
    
    bc_tip_x = fem.dirichletbc(u_twist_x, dof_indices_x_top, V_x)
    bc_tip_z = fem.dirichletbc(u_twist_z, dof_indices_z_top, V_z)

    bcs =[bc_fixed, bc_comp, bc_tip_x, bc_tip_z]

    u_xdmf, stress_xdmf, strain_xdmf, uh = _solve_and_write(msh, V, bcs, filesave_path)
    return u_xdmf, stress_xdmf, strain_xdmf, bcs, uh

# ============================================================
# IO helpers
# ============================================================

class XDMFData:
    def __init__(self, coords=None, topo=None, fields=None):
        self.coords = coords
        self.topo   = topo
        self.fields = fields or {}


def load_xdmf(xmf_path):
    xmf_path = os.path.abspath(xmf_path)
    xmf_dir  = os.path.dirname(xmf_path)
    tree = ET.parse(xmf_path)
    root = tree.getroot()
    coords, topo, fields = None, None, {}
    for dataitem in root.findall(".//DataItem"):
        text = dataitem.text.strip()
        if ":" not in text:
            continue
        h5_file, h5_dataset = text.split(":", 1)
        h5_path = os.path.join(xmf_dir, h5_file)
        dims = tuple(map(int, dataitem.get("Dimensions").split()))
        parent_tag = dataitem.getparent().tag if hasattr(dataitem, "getparent") else None
        name = None
        if parent_tag == "Geometry":
            name = "coords"
        elif parent_tag == "Topology":
            name = "topology"
        else:
            attribute = dataitem.find("..")
            name = attribute.get("Name") if attribute is not None else None
        with h5py.File(h5_path, "r") as h5:
            arr = h5[h5_dataset][:]
        if name == "coords":
            coords = arr.reshape(dims)
        elif name == "topology":
            topo = arr.reshape(dims)
        elif name is not None:
            fields[name] = arr.reshape(dims)
    return XDMFData(coords=coords, topo=topo, fields=fields)


def load_solution(xdmf_path, mesh_xdmf_path=None):
    comm = MPI.COMM_WORLD
    if mesh_xdmf_path is None:
        mesh_xdmf_path = xdmf_path

    with io.XDMFFile(comm, mesh_xdmf_path, "r") as xdmf:
        try:
            mesh_obj = xdmf.read_mesh(name="mesh")
        except RuntimeError:
            mesh_obj = xdmf.read_mesh(name="Grid")

    element = basix.ufl.element(
        "Lagrange", mesh_obj.basix_cell(), 1,
        shape=(mesh_obj.geometry.dim,))
    V = fem.functionspace(mesh_obj, element)

    xmf_dir = os.path.dirname(os.path.abspath(xdmf_path))
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    h5_file = h5_dataset = dims = None
    for attr in root.findall(".//Attribute"):
        if attr.get("Name") == "u":
            dataitem = attr.find("DataItem")
            text = dataitem.text.strip()
            h5_file, h5_dataset = text.split(":", 1)
            dims = tuple(map(int, dataitem.get("Dimensions").split()))
            break

    if h5_file is None:
        raise RuntimeError("Could not find Attribute Name='u' in XDMF")

    with h5py.File(os.path.join(xmf_dir, h5_file), "r") as h5:
        u_data = h5[h5_dataset][:].reshape(dims)

    u = fem.Function(V)
    u.x.array[:] = u_data.reshape(-1)
    return mesh_obj, V, u


# ============================================================
# ParaView interpolation
# ============================================================

PV = "./ParaView-5.12.0-MPI-Linux-Python3.10-x86_64/bin/pvpython"
PV_SCRIPT = "./paraview_compare.py"


def interpolate_to_ref_mesh_paraview(src, dst, experiment_name=None):
    out = f"{src.split('.xdmf')[0]}~interpolated"
    if experiment_name:
        out += f"_{experiment_name}"
    out += ".xdmf"
    cmd = [PV, PV_SCRIPT, src, dst, out, experiment_name or ""]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env={"LD_LIBRARY_PATH": ""})
    out_text, err_text = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"pvpython failed with exit {proc.returncode}")
    return out


# ============================================================
# Error computation
# ============================================================

def compute_disp_errors_on_ref(ref_xdmf, cand_xdmf, E=210e9, nu=0.3,
                                experiment_name=None):
    mesh_ref, V_ref, _ = load_solution(ref_xdmf)

    u_cand_interp_path = interpolate_to_ref_mesh_paraview(cand_xdmf, ref_xdmf, experiment_name)
    u_ref_interp_path  = interpolate_to_ref_mesh_paraview(ref_xdmf,  ref_xdmf, experiment_name)

    *_, u_cand_interp = load_solution(u_cand_interp_path, ref_xdmf)
    *_, u_ref_interp  = load_solution(u_ref_interp_path,  ref_xdmf)

    diff = fem.Function(V_ref)
    diff.x.array[:] = u_cand_interp.x.array - u_ref_interp.x.array

    L2_err = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)))
    L2_ref = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_ref_interp, u_ref_interp) * ufl.dx)))

    mu     = E / (2.0 * (1.0 + nu))
    lmbda  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def psi(u):
        e = eps(u)
        return lmbda * ufl.tr(e)**2 + 2.0 * mu * ufl.inner(e, e)

    energy_err = np.sqrt(abs(fem.assemble_scalar(fem.form(0.5 * psi(diff) * ufl.dx))))
    energy_ref = np.sqrt(abs(fem.assemble_scalar(fem.form(0.5 * psi(u_ref_interp) * ufl.dx))))

    return {
        "L2_err":    float(L2_err),
        "rel_L2":    float(L2_err / (L2_ref + 1e-30)),
        "energy_err": float(energy_err),
        "rel_energy": float(energy_err / (energy_ref + 1e-30)),
    }


# ============================================================
# Interpolation-free scalar QoI errors (rebuttal tables)
# ============================================================

import re as _re


def read_pos_points_top(pos_file, top_percentile=99.9, load_case=None,
                        remove_boundary_frac=0.02):
    """Read SP points from a gmsh .pos file, keep top percentile by value.

    If load_case is given, boundary/load singularities are trimmed first
    using the same convention as results_analysis_rebuttal.read_pos_points
    (drop points within remove_boundary_frac of the bbox extremes along
    the loading axes) before the percentile cut.
    Returns (points (M,3), values (M,))."""
    pattern = _re.compile(r"SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};")
    points, values = [], []
    with open(pos_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                x, y, z, val = map(float, m.groups())
                points.append([x, y, z])
                values.append(val)
    points, values = np.asarray(points), np.asarray(values)
    if len(values) == 0:
        return np.empty((0, 3)), np.empty((0,))
    if load_case is not None:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        x_dist = (x_max - x_min) * remove_boundary_frac
        y_dist = (y_max - y_min) * remove_boundary_frac
        mask = np.ones(len(points), dtype=bool)
        if load_case in ("compression", "bending", "torsion", "shear"):
            mask &= (points[:, 1] < y_max - y_dist) & (points[:, 1] > y_min + y_dist)
        elif load_case in ("bending_compression", "torsion_compression"):
            mask &= (points[:, 0] < x_max - x_dist) & (points[:, 0] > x_min + x_dist)
            mask &= points[:, 1] < y_max - y_dist
        points, values = points[mask], values[mask]
        if len(values) == 0:
            return np.empty((0, 3)), np.empty((0,))
    cutoff = np.percentile(values, top_percentile)
    mask = values >= cutoff
    return points[mask], values[mask]


def compute_qoi_scalars(sol_xdmf, E=210e9, nu=0.3,
                        crit_points=None, crit_radius=None):
    """Interpolation-free scalar QoIs evaluated natively on the mesh of
    `sol_xdmf` (a *_sol.xdmf displacement solution).

    Since u is CG1, eps/sigma are exactly constant per cell, so the
    cell-wise von Mises and energy density values are exact (no
    projection/interpolation error). Omega_crit = cells whose centroid
    lies within crit_radius of any crit_point (e.g. top-percentile ZZ
    points of the fine reference).

    Returns dict with: n_dofs, n_cells, energy_total, energy_crit,
    vm_p99_crit, vm_max_crit, max_disp, n_crit_cells.
    """
    from scipy.spatial import cKDTree

    msh, V, u = load_solution(sol_xdmf)
    tdim = msh.topology.dim
    n_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    n_cells = msh.topology.index_map(tdim).size_local

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    e = eps(u)
    s = 2.0 * mu * e + lmbda * ufl.tr(e) * ufl.Identity(3)
    psi = 0.5 * ufl.inner(s, e)                      # strain energy density
    dev = s - (ufl.tr(s) / 3.0) * ufl.Identity(3)
    vm = ufl.sqrt(1.5 * ufl.inner(dev, dev) + 1e-30)  # von Mises

    # cell-wise exact values via DG0 assembly (integrand constant per cell)
    W0 = fem.functionspace(msh, ("DG", 0))
    v0 = ufl.TestFunction(W0)
    cell_vols = fem.assemble_vector(fem.form(v0 * ufl.dx)).array.copy()
    psi_cell = fem.assemble_vector(fem.form(psi * v0 * ufl.dx)).array / cell_vols
    vm_cell = fem.assemble_vector(fem.form(vm * v0 * ufl.dx)).array / cell_vols

    energy_total = float(psi_cell @ cell_vols)
    max_disp = float(np.linalg.norm(
        u.x.array.reshape(-1, msh.geometry.dim), axis=1).max())

    res = {
        "n_dofs": int(n_dofs), "n_cells": int(n_cells),
        "energy_total": energy_total, "max_disp": max_disp,
        "volume_total": float(cell_vols.sum()),
        "energy_crit": float("nan"), "vm_p99_crit": float("nan"),
        "vm_max_crit": float("nan"), "n_crit_cells": 0,
    }

    if crit_points is not None and len(crit_points) > 0 and crit_radius:
        centroids = msh.geometry.x[msh.geometry.dofmap].mean(axis=1)
        d, _ = cKDTree(np.asarray(crit_points, float)).query(centroids)
        crit = d <= crit_radius
        n_crit = int(crit.sum())
        res["n_crit_cells"] = n_crit
        if n_crit > 0:
            res["energy_crit"] = float(psi_cell[crit] @ cell_vols[crit])
            res["vm_p99_crit"] = float(np.percentile(vm_cell[crit], 99))
            res["vm_max_crit"] = float(vm_cell[crit].max())

    return res


# ============================================================
# Top-level dispatcher
# ============================================================

_SOLVERS = {
    "compression":         run_fenics_compression,
    "bending":             run_fenics_bending,
    "torsion":             run_fenics_torsion,
    "shear":               run_fenics_shear,
    "bending_compression": run_fenics_bending_compression,
    "torsion_compression": run_fenics_torsion_compression,
}

def detect_left_right_facets(msh, facet_normals, x_min, x_max, cos_thresh=0.3, tol=1e-3):
    """Detect facets whose outward normal is ±x (used for fixed supports and X-compression)."""
    from dolfinx import mesh as dmesh_mod
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    conn = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x
    
    boundary_facets = dmesh_mod.exterior_facet_indices(msh.topology)
    left_facets, right_facets = [],[]
    
    for f in boundary_facets:
        verts = conn.links(f)
        x_verts = coords[verts, 0]
        n_x = np.abs(facet_normals[f, 0])
        
        if n_x >= cos_thresh:
            if np.all(np.abs(x_verts - x_min) < tol):
                left_facets.append(f)
            elif np.all(np.abs(x_verts - x_max) < tol):
                right_facets.append(f)
                
    print(f"[detect_left_right_facets] Boundary: {len(boundary_facets)} "
          f"| left (-x): {len(left_facets)} | right (+x): {len(right_facets)}")
    return np.array(left_facets, dtype=np.int32), np.array(right_facets, dtype=np.int32)

def compute_disp_error(candidate_mshs, ref_msh,
                       outfile="results_fenics.log",
                       E=210e9, nu=0.3,
                       solve_reference=True,
                       experiment_name=None,
                       load_case="compression",
                       **solver_kwargs):
    if load_case not in _SOLVERS:
        raise ValueError(f"Unknown load_case '{load_case}'.")

    solver_fn = _SOLVERS[load_case]
    solved_uh = {}

    if solve_reference:
        ref_u_xdmf, *_, uh_ref = solver_fn(ref_msh, E=E, nu=nu, **solver_kwargs)
        solved_uh['ref'] = uh_ref
    else:
        ref_u_xdmf = f"{ref_msh.split('.msh')[0]}_sol.xdmf"
        solved_uh['ref'] = ref_u_xdmf

    res_cand = {}
    for msh_path in candidate_mshs:
        suffix = msh_path.split(".msh")[0].split("/")[-1].split("_")[0]
        u_xdmf, stress_xdmf, strain_xdmf, bcs_cand, uh_cand = solver_fn(
            msh_path, E=E, nu=nu, **solver_kwargs)

        res_suffix = compute_disp_errors_on_ref(
            ref_u_xdmf, u_xdmf, E=E, nu=nu, experiment_name=experiment_name)

        if outfile is not None:
            os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
            with open(outfile, "a") as fout:
                fout.write(f"[{load_case}] {msh_path.split('/')[-1]}: {res_suffix}\n")

        print(res_suffix)
        res_cand[suffix] = res_suffix

    return solved_uh, res_cand


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--meshA",   required=True)
    p.add_argument("--meshB",   default=None)
    p.add_argument("--meshRef", required=True)
    p.add_argument("--outdir",  default="results_fenics")
    p.add_argument("--E",       type=float, default=210e9)
    p.add_argument("--nu",      type=float, default=0.3)
    p.add_argument("--load_case", default="compression", choices=["compression", "bending", "torsion"])
    p.add_argument("--disp_frac",       type=float, default=0.01)
    p.add_argument("--bend_axis",       type=int,   default=1)
    p.add_argument("--twist_angle_deg", type=float, default=1.0)
    args = p.parse_args()

    solver_kwargs = {}
    if args.load_case == "compression":
        solver_kwargs["disp_frac"] = args.disp_frac
    elif args.load_case == "bending":
        solver_kwargs["disp_frac"]  = args.disp_frac
        solver_kwargs["bend_axis"]  = args.bend_axis
    elif args.load_case == "torsion":
        solver_kwargs["twist_angle_deg"] = args.twist_angle_deg

    candidates = [args.meshA]
    if args.meshB: candidates.append(args.meshB)

    outfile = os.path.join(args.outdir, f"results_{args.load_case}.log")

    compute_disp_error(
        candidates, args.meshRef,
        outfile=outfile,
        E=args.E, nu=args.nu,
        load_case=args.load_case,
        **solver_kwargs
    )