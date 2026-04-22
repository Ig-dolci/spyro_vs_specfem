#!/usr/bin/env python3
"""Compute KMV h values that give similar DOF counts to spectral defaults.

Run: python compute_kmv_equivalent_hs.py

Creates only 4 small test meshes to measure DOF density, then extrapolates.
"""
import firedrake


def scalar_dofs(mesh, family, degree):
    V = firedrake.FunctionSpace(mesh, family, degree)
    return V.dof_count


# ---- Measure DOF density from one small mesh each ----
print("Measuring DOF density from small test meshes...\n")

n2 = 5
L2 = 3.0
mesh_q = firedrake.RectangleMesh(n2, n2, L2, L2, quadrilateral=True)
dofs_spec_2d = scalar_dofs(mesh_q, "CG", 4)
mesh_t = firedrake.RectangleMesh(n2, n2, L2, L2, quadrilateral=False)
dofs_kmv_2d = scalar_dofs(mesh_t, "KMV", 4)
print(f"2D n={n2}: spectral CG4={dofs_spec_2d}, KMV4={dofs_kmv_2d}, ratio={dofs_kmv_2d/dofs_spec_2d:.3f}")

n3 = 4
L3 = 1.0
q2d = firedrake.RectangleMesh(n3, n3, L3, L3, quadrilateral=True)
mesh_hex = firedrake.ExtrudedMesh(q2d, n3, layer_height=L3/n3)
dofs_spec_3d = scalar_dofs(mesh_hex, "CG", 4)
mesh_tet = firedrake.BoxMesh(n3, n3, n3, L3, L3, L3)
dofs_kmv_3d = scalar_dofs(mesh_tet, "KMV", 3)
print(f"3D n={n3}: spectral CG4={dofs_spec_3d}, KMV3={dofs_kmv_3d}, ratio={dofs_kmv_3d/dofs_spec_3d:.3f}")

# DOFs ~ alpha * n^dim.  To match: n_k = n_s * (alpha_s/alpha_k)^(1/dim)
a_s2 = dofs_spec_2d / n2**2
a_k2 = dofs_kmv_2d / n2**2
scale_2d = (a_s2 / a_k2) ** 0.5

a_s3 = dofs_spec_3d / n3**3
a_k3 = dofs_kmv_3d / n3**3
scale_3d = (a_s3 / a_k3) ** (1.0/3)

print(f"\n2D scale={scale_2d:.4f}  =>  h_kmv ≈ h_spec * {1/scale_2d:.3f}")
print(f"3D scale={scale_3d:.4f}  =>  h_kmv ≈ h_spec * {1/scale_3d:.3f}")


def nearest_h(L, n_float):
    n = max(1, round(n_float))
    return L / n, n


# ---- 2D ----
print("\n" + "=" * 70)
print("2D: L=3.0, spectral p=4, KMV p=4")
print("=" * 70)
spec_hs = [0.12, 0.1, 0.075, 0.06, 0.05, 0.04, 0.03, 0.02]
spec_ref = 0.015

print(f"{'h_spec':>8} {'n_s':>5} {'DOF_s':>9} | {'h_kmv':>10} {'n_k':>5} {'DOF_k':>9} {'ratio':>7}")
sweep_2d = []
for h_s in spec_hs + [spec_ref]:
    n_s = round(L2 / h_s)
    dof_s = a_s2 * n_s**2
    h_k, n_k = nearest_h(L2, n_s * scale_2d)
    dof_k = a_k2 * n_k**2
    tag = " (ref)" if h_s == spec_ref else ""
    print(f"{h_s:8.4f} {n_s:5d} {dof_s:9.0f} | {h_k:10.6f} {n_k:5d} {dof_k:9.0f} {dof_k/dof_s:7.3f}{tag}")
    if h_s != spec_ref:
        sweep_2d.append(h_k)
    else:
        ref_2d = h_k

print(f"\nDEFAULT_SPACE_HS_KMV = {sweep_2d}")
print(f"DEFAULT_REFERENCE_H_KMV = {ref_2d}")

# ---- 3D ----
print("\n" + "=" * 70)
print("3D: L=1.0, spectral p=4, KMV p=3")
print("=" * 70)
spec_hs3 = [0.125, 0.1, 0.0625, 0.05]
spec_ref3 = 0.04

print(f"{'h_spec':>8} {'n_s':>5} {'DOF_s':>9} | {'h_kmv':>10} {'n_k':>5} {'DOF_k':>9} {'ratio':>7}")
sweep_3d = []
for h_s in spec_hs3 + [spec_ref3]:
    n_s = round(L3 / h_s)
    dof_s = a_s3 * n_s**3
    h_k, n_k = nearest_h(L3, n_s * scale_3d)
    dof_k = a_k3 * n_k**3
    tag = " (ref)" if h_s == spec_ref3 else ""
    print(f"{h_s:8.4f} {n_s:5d} {dof_s:9.0f} | {h_k:10.6f} {n_k:5d} {dof_k:9.0f} {dof_k/dof_s:7.3f}{tag}")
    if h_s != spec_ref3:
        sweep_3d.append(h_k)
    else:
        ref_3d = h_k

print(f"\nDEFAULT_SPACE_HS_KMV = {sweep_3d}")
print(f"DEFAULT_REFERENCE_H_KMV = {ref_3d}")
