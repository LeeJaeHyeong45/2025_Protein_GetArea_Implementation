# getArea.py
# Analytical ASA (Accessible Surface Area) via Gauss–Bonnet using IHS + geometric inversion.
# Based on Fraczkiewicz & Braun (1998) GETAREA routine: intersection-of-half-spaces (IHS),
# geometric inversion, convex hull, and Gauss–Bonnet path over solvent-accessible arcs.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from re import S
from tkinter import NO
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math
import numpy as np
from scipy.spatial import ConvexHull
import csv
import os
import argparse
import sys
from scipy.optimize import linprog

# ----------------------------
# 0) Hyperparameters & types
# ----------------------------

PROBE_RADIUS = 1.4  # Å (Richards), ASA radius = vdW + probe

class AtomClass(Enum):
    ALIPHATIC_C = "aliphatic_C"
    AROMATIC_C = "aromatic_C"
    CARBONYL_C = "carbonyl_C"
    AMIDE_N = "amide_N"
    AMINE_N = "amine_N"
    CARBONYL_O = "carbonyl_O"
    HYDROXYL_O = "hydroxyl_O"
    THIOL_S = "thiol_S"
    SULFUR_S = "sulfur_S"
    OTHER = "other"

@dataclass
class AtomType:
    klass: AtomClass
    polar: bool
    vdw_radius: float  # vdW radius (Å)
    asa_radius: float  # = vdw_radius + PROBE_RADIUS

# Minimal starting radii (you can swap to Ooi/S&R sets later).
# Table I in the paper lists two vdW sets; we start with something practical & adjust later. :contentReference[oaicite:7]{index=7}
DEFAULT_ATOM_TYPES: Dict[AtomClass, AtomType] = {
    AtomClass.ALIPHATIC_C: AtomType(AtomClass.ALIPHATIC_C, False, 2.00, 2.00 + PROBE_RADIUS),
    AtomClass.AROMATIC_C:  AtomType(AtomClass.AROMATIC_C,  False, 1.75, 1.75 + PROBE_RADIUS),
    AtomClass.CARBONYL_C:  AtomType(AtomClass.CARBONYL_C,  False, 1.55, 1.55 + PROBE_RADIUS),
    AtomClass.AMIDE_N:     AtomType(AtomClass.AMIDE_N,     True,  1.55, 1.55 + PROBE_RADIUS),
    AtomClass.AMINE_N:     AtomType(AtomClass.AMINE_N,     True,  1.55, 1.55 + PROBE_RADIUS),
    AtomClass.CARBONYL_O:  AtomType(AtomClass.CARBONYL_O,  True,  1.40, 1.40 + PROBE_RADIUS),
    AtomClass.HYDROXYL_O:  AtomType(AtomClass.HYDROXYL_O,  True,  1.40, 1.40 + PROBE_RADIUS),
    AtomClass.THIOL_S:     AtomType(AtomClass.THIOL_S,     True,  2.00, 2.00 + PROBE_RADIUS),
    AtomClass.SULFUR_S:    AtomType(AtomClass.SULFUR_S,    True,  2.00, 2.00 + PROBE_RADIUS),
    AtomClass.OTHER:       AtomType(AtomClass.OTHER,       True,  0.0, 0.0),
}

@dataclass
class Atom:
    idx: int
    name: str
    res_name: str       # Added: Residue Name (e.g., PRO)
    res_seq: int        # Added: Residue Sequence (e.g., 2)
    element: str
    coord: np.ndarray
    type: AtomType


@dataclass
class Molecule:
    atoms: List[Atom]


# Type aliases for readability
Vec3 = np.ndarray
AtomIndex = int
PlaneIndex = int
EdgeKey = Tuple[PlaneIndex, PlaneIndex] # 2 plane indices (k, l), NOT SORTED.
VertexIndex = int


# ---------------------------------------
# 1) Very light PDB parser (ATOM/HETATM)
# ---------------------------------------

def guess_atom_class(element: str, name: str) -> AtomClass:
    # Simple, conservative heuristics. You can replace with a chem-perception library later.
    e = element.upper()
    nm = name.upper()
    if e == "C":
        if nm.startswith("C") and ("A" in nm or "B" in nm):   # crude; tweak later
            return AtomClass.AROMATIC_C
        if "O" in nm or "OXT" in nm:
            return AtomClass.CARBONYL_C
        return AtomClass.ALIPHATIC_C
    if e == "N":
        if "H" in nm or "AM" in nm:
            return AtomClass.AMINE_N
        return AtomClass.AMIDE_N
    if e == "O":
        if "OH" in nm or nm.startswith("O") and not nm.endswith("XT"):
            return AtomClass.HYDROXYL_O
        return AtomClass.CARBONYL_O
    if e == "S":
        if "H" in nm:
            return AtomClass.THIOL_S
        return AtomClass.SULFUR_S
    return AtomClass.OTHER

def parse_pdb(path: str,
              atom_types: Dict[AtomClass, AtomType] = DEFAULT_ATOM_TYPES) -> Molecule:
    atoms: List[Atom] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        idx = 0
        for line in f:
            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            
            # Standard PDB Column Parsing
            name = line[12:16].strip()
            alt_loc = line[16] # Alternate location indicator
            res_name = line[17:20].strip()
            
            # Skip alternate locations other than ' ' or 'A' to avoid duplicates
            if alt_loc not in (' ', 'A'):
                continue

            try:
                # Parse coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                # Parse residue sequence (handle potential hex or errors)
                res_seq_str = line[22:26].strip()
                res_seq = int(res_seq_str) if res_seq_str else 0
                
            except ValueError:
                continue

            element = (line[76:78].strip() or name[0]).upper()
            klass = guess_atom_class(element, name)
            atype = atom_types[klass]
            
            atom = Atom(idx=idx, 
                        name=name, 
                        res_name=res_name, 
                        res_seq=res_seq, 
                        element=element,
                        coord=np.array([x, y, z], dtype=float),
                        type=atype)
            atoms.append(atom)
            idx += 1
    return Molecule(atoms=atoms)


# --------------------------------------------
# 2) Spatial hash grid (cubic lattice, 27 cells)
# --------------------------------------------

class SpatialHash:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.table: Dict[Tuple[int, int, int], List[int]] = {}

    def _key(self, p: np.ndarray) -> Tuple[int, int, int]:
        return tuple((p // self.cell_size).astype(int))  # floor division per axis

    def build(self, mol: Molecule):
        self.table.clear()
        for a in mol.atoms:
            k = self._key(a.coord)
            self.table.setdefault(k, []).append(a.idx)

    def neighbors(self, mol: Molecule, center_idx: int) -> Iterable[int]:
        """Yield candidate neighbor indices within 27 cells (3x3x3)."""
        p = mol.atoms[center_idx].coord
        kx, ky, kz = self._key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    kk = (kx + dx, ky + dy, kz + dz)
                    for j in self.table.get(kk, []):
                        if j != center_idx:
                            yield j


# -------------------------------------------------------
# 3) Geometry helpers: planes, inversion, intersections
# -------------------------------------------------------

@dataclass
class Plane:
    """
    Half-space defined in the local frame of the central atom: n · (x - x_i) = g.
    The unit normal n points toward the neighbor atom; g is the scalar offset.
    """
    n: Vec3
    g: float

def unit(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Zero-length vector.")
    return v / n

def circle_plane_for_neighbor(x_i: Vec3, r_i: float,
                              x_k: Vec3, r_k: float) -> Optional[Plane]:
    """
    Plane of the circle of intersection (COI) on sphere i induced by neighbor k.
    Paper notation: m_k = (x_k - x_i)/d,  g_k = (d^2 + r_i^2 - r_k^2) / (2d).  (eq. 6) :contentReference[oaicite:8]{index=8}
    We return half-space H_k:  m_k · (x - x_i) <= g_k  (accessible side).
    If spheres do not intersect (|r_i - r_k| <= d <= r_i + r_k), we still get a plane:
      - If d >= r_i + r_k: k is too far; has no effect → return None.
      - If d <= |r_i - r_k| and r_k >= r_i: i fully buried by k → return None (no ASA).
      - If d <= |r_i - r_k| and r_k < r_i: k fully inside i → still defines a cap; keep plane.
      TODO: r_k is VERY large enough to bury this from the inside
    """
    dvec = x_k - x_i # from this point, we treat x_i as origin
    d = np.linalg.norm(dvec)
    if d == 0.0:
        return None
    m_k = dvec / d 
    if d <= abs(r_i - r_k) and r_k >= r_i:
        return None
    g_k = (d*d + r_i*r_i - r_k*r_k) / (2.0*d)  # eq. (6) --> may be negative if the neighbor is very large
    # Simple culling: if neighbor too far to touch the central sphere, skip.
    if d >= (r_i + r_k):
        return None
    return Plane(n=m_k, g=g_k) # g may be negative, while m_k ALWAYS TOWARD NEIGHBOR ATOM CENTER - this is for information of center way, but needs to be addressed later for vector calculations

def invert_vector(v: np.ndarray) -> np.ndarray:
    """
    Geometric inversion in unit sphere: v -> v / ||v||^2  (dual space).
    (Fig. 4 in the paper; we invert the distance vectors to COI planes.) :contentReference[oaicite:9]{index=9}
    """
    n2 = np.dot(v, v)
    if n2 == 0.0:
        return v.copy()
    return v / n2

def triple_plane_intersection(p1: Plane, p2: Plane, p3: Plane) -> Optional[np.ndarray]:
    """
    Solve for x such that:
      n1 · (x - 0) = g1,  n2 · x = g2,  n3 · x = g3
    Return x (IHS vertex in primal/original space relative to x_i), or None if degenerate.
    """
    N = np.vstack([p1.n, p2.n, p3.n])  # 3x3
    g = np.array([p1.g, p2.g, p3.g])
    try:
        x = np.linalg.solve(N, g)
        return x
    except np.linalg.LinAlgError:
        return None


# ------------------------------------------------------
# 4) Dual convex hull and mapping back to IHS topology
# ------------------------------------------------------

@dataclass
class DualPoint:
    neighbor_index: int        # which neighbor plane generated this dual point
    dual_coord: np.ndarray     # inverted vector p_k = g_k * m_k, then invert

class PlaneLoopType(Enum):
    """Qualitative classification of the circle/arc traced on plane k."""
    DEGENERATE = auto()      # no usable edges (plane not active)
    CLOSED = auto()          # convex polygon (full loop)
    OPEN = auto()            # open chain (cone-like)
    PARALLEL = auto()        # parallel-plane case (points at infinity)

@dataclass
class IHSPlane:
    plane_index: PlaneIndex # is ALWAYS EQUAL TO CORRESPONDING NEIGHBOR ATOM
    unit_vec: Vec3
    offset: float
    mu_vector: Vec3  # x_k - x_i unit vector.
    vertex_order: List[VertexIndex] = field(default_factory=list)
    edge_order: List[EdgeKey] = field(default_factory=list) # list of edge keys(= plane tuples), NOT SORTED.
    edge_vertex_pairs: List[Tuple[VertexIndex, VertexIndex]] = field(default_factory=list) # list of vertex index pairs for each edge in order
    vertex_inside_flags: List[bool] = field(default_factory=list)
    loop_type: PlaneLoopType = PlaneLoopType.DEGENERATE
    orientation_sign: int = 0  # -1 if (e1 x e2)·mu < 0, +1 otherwise

@dataclass
class IHSVertex:
    planes: Tuple[PlaneIndex, PlaneIndex, PlaneIndex]
    coord: Vec3
    inside_sphere: bool

@dataclass
class IHSEdge:
    planes_pair: EdgeKey
    vertex_ids: Tuple[VertexIndex, VertexIndex]

    def other_plane(self, plane_id: PlaneIndex) -> PlaneIndex:
        k, l = self.planes_pair
        if plane_id == k:
            return l
        if plane_id == l:
            return k
        raise ValueError(f"Plane {plane_id} not incident to edge {self.planes_pair}.")

    def other_vertex(self, vertex_id: VertexIndex) -> VertexIndex:
        v1, v2 = self.vertex_ids
        if vertex_id == v1:
            return v2
        if vertex_id == v2:
            return v1
        raise ValueError(f"Vertex {vertex_id} not part of edge {self.vertex_ids}.")

@dataclass
class IHS:
    planes: List[IHSPlane]
    vertices: List[IHSVertex]
    edges: List[IHSEdge]


def _coords_from_sequence(vertex_ids: Sequence[VertexIndex],
                          vertices: Sequence[IHSVertex]) -> Optional[List[Vec3]]: # should change -> allow infinity for cone and parallel
    coords: List[Vec3] = []
    for vid in vertex_ids:
        coord = vertices[vid].coord
        if not np.all(np.isfinite(coord)):
            return None
        coords.append(coord)
    return coords


def _orientation_metric(coords: Sequence[np.ndarray],
                        mu_dir: np.ndarray,
                        closed: bool) -> float:
    """
    Calculates alignment score using the Area Vector (cross product of 2 edge vectors).
    This handles collinear vertices and offset origins robustly.
    """
    n_pts = len(coords)
    if n_pts < 3:
        return 0.0

    # Calculate Area Vector (Normal of the polygon)
    area_vec = np.zeros(3, dtype=float)
    

    p_3 = coords[2]
    p_2 = coords[1]
    p_1 = coords[0]

    e_1 = p_2 - p_1
    e_2 = p_3 - p_2

    area_vec = np.cross(e_1, e_2)

    # Score: Dot product with reference normal
    return float(np.dot(area_vec, mu_dir))

def _orient_vertex_sequence(plane: IHSPlane,
                            vertex_seq: List[VertexIndex],
                            vertices: Sequence[IHSVertex]) -> Tuple[List[VertexIndex], int]:
    if len(vertex_seq) < 3:
        return vertex_seq, 0
    
    mu_norm = float(np.linalg.norm(plane.mu_vector))
    if mu_norm == 0.0:
        return vertex_seq, 0
    
    # Get coordinates
    coords = []
    for v_idx in vertex_seq:
        if v_idx >= len(vertices): return vertex_seq, 0
        v = vertices[v_idx]
        if np.isinf(v.coord[0]): return vertex_seq, 0
        coords.append(v.coord)
        
    mu_dir = plane.mu_vector / mu_norm
    closed = plane.loop_type == PlaneLoopType.CLOSED
    
    score = _orientation_metric(coords, mu_dir, closed)
    
    # If score < 0, it means the Area Vector opposes the Normal Vector.
    # This implies the winding is "Clockwise" relative to the Normal.
    # Depending on convention, we flip. Standard right-hand rule says CCW aligns with Normal.
    # If your system expects CCW, and we got negative score, we flip.
    
    if score < -1e-9:
        return list(reversed(vertex_seq)), -1
    elif score > 1e-9:
        # Aligned
        return vertex_seq, 0
    else:
        # Ambiguous
        return vertex_seq, 0

def build_dual_points(planes: List[Plane]) -> List[DualPoint]:
    dual: List[DualPoint] = []
    for idx, pl in enumerate(planes):
        # distance vector from origin to plane: d_k = g_k * n_k
        dvec = pl.g * pl.n
        dual.append(DualPoint(neighbor_index=idx, dual_coord=invert_vector(dvec)))
    # Add "plane at infinity" → maps to origin
    dual.append(DualPoint(neighbor_index=len(planes), dual_coord=np.array([0.0, 0.0, 0.0], dtype=float)))
    return dual

def convex_hull_on_dual(dual_points: List[DualPoint]) -> ConvexHull:
    pts = np.vstack([dp.dual_coord for dp in dual_points])  # Nx3
    return ConvexHull(pts)  # qhull in 3D; faces in hull.simplices (triangles)

def hull_faces_to_ihs_vertices(planes: List[Plane],
                               hull: ConvexHull,
                               sphere_radius: float) -> List[IHSVertex]:
    """
    Robust mapping of Dual Hull Faces -> Primal IHS Vertices.
    Ignores faces that involve the 'Plane at Infinity' (index = len(planes)).
    """
    vertices: List[IHSVertex] = []
    infinity_index = len(planes)
    
    for tri in hull.simplices:  # shape (3,)
        # Check if this face touches the infinity point (dual of origin/box)
        if any(t == infinity_index for t in tri):
            continue
            
        k, j, l = (int(t) for t in tri)
        
        # Verify indices are valid
        if k >= len(planes) or j >= len(planes) or l >= len(planes):
            continue

        p = triple_plane_intersection(planes[k], planes[j], planes[l])
        plane_triple = tuple(sorted((k, j, l)))
        
        if p is None: 
            # Vertices at infinity are generally not part of the SASA closed loop
            # unless we are clipping with a bounding box (which we are).
            # If p is None here, it means planes are parallel.
            continue
            
        # Robust Inside Check with epsilon
        dist_sq = float(np.dot(p, p))
        inside_flag = dist_sq < (sphere_radius * sphere_radius + 1e-9)
        
        vertices.append(IHSVertex(planes=plane_triple, coord=p, inside_sphere=inside_flag))
        
    return vertices

def build_ihs_edges_from_face_adjacency(vertices: List[IHSVertex]) -> List[IHSEdge]:
    """
    Two IHS vertices are connected by an edge iff their plane triples share exactly 2 planes.
    The edge is labeled by that shared plane pair (k, l) with k<l for canonical ordering.
    """
    edges: List[IHSEdge] = []
    # Compare all pairs (O(V^2) is fine; V per central atom is modest)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            s1 = set(vertices[i].planes)
            s2 = set(vertices[j].planes)
            inter = s1.intersection(s2)
            if len(inter) == 2:
                k, l = sorted(inter)
                edges.append(IHSEdge(planes_pair=(k, l), vertex_ids=(i, j)))
    return edges


def build_plane_topology(valid_planes: List[Plane],
                         ihs_vertices: List[IHSVertex],
                         ihs_edges: List[IHSEdge]) -> List[IHSPlane]:
    # 1. Initialization
    planes: List[IHSPlane] = []
    for idx, plane in enumerate(valid_planes):
        planes.append(IHSPlane(
            plane_index=idx,
            unit_vec=plane.n,
            offset=plane.g,
            mu_vector=plane.n,
            loop_type=PlaneLoopType.CLOSED 
        ))

    if not ihs_edges: return planes

    # 2. Build Adjacency
    edges_by_plane: Dict[PlaneIndex, List[int]] = {}
    adjacency_by_plane: Dict[PlaneIndex, Dict[VertexIndex, List[int]]] = {}
    edge_lookup_by_plane: Dict[PlaneIndex, Dict[Tuple[VertexIndex, VertexIndex], int]] = {}

    for edge_idx, edge in enumerate(ihs_edges):
        k, l = edge.planes_pair
        v1, v2 = edge.vertex_ids
        for plane_id in (k, l):
            edges_by_plane.setdefault(plane_id, []).append(edge_idx)
            adj = adjacency_by_plane.setdefault(plane_id, {})
            adj.setdefault(v1, []).append(edge_idx)
            adj.setdefault(v2, []).append(edge_idx)
            key = (min(v1, v2), max(v1, v2))
            edge_lookup_by_plane.setdefault(plane_id, {})[key] = edge_idx

    # 3. Build Cycles
    for plane in planes:
        plane_id = plane.plane_index
        edge_indices = edges_by_plane.get(plane_id, [])
        if not edge_indices: continue 

        adjacency = adjacency_by_plane[plane_id]
        start_edge = edge_indices[0]
        start_vertex = ihs_edges[start_edge].vertex_ids[0]
        
        vertex_sequence: List[VertexIndex] = []
        visited_edges: Set[int] = set()
        
        curr_v = start_vertex
        curr_e = start_edge
        
        while True:
            if not vertex_sequence or vertex_sequence[-1] != curr_v:
                vertex_sequence.append(curr_v)
            visited_edges.add(curr_e)
            edge = ihs_edges[curr_e]
            next_v = edge.other_vertex(curr_v)
            
            candidates = adjacency.get(next_v, [])
            next_e = None
            for cand in candidates:
                if cand != curr_e:
                    next_e = cand; break
            if next_e is None or next_v == start_vertex: break
            curr_v = next_v
            curr_e = next_e

        # Initial Orientation
        oriented, sign = _orient_vertex_sequence(plane, vertex_sequence, ihs_vertices)
        plane.vertex_order = oriented
        plane.orientation_sign = sign
        plane.vertex_inside_flags = [ihs_vertices[v].inside_sphere for v in plane.vertex_order]

    # 4. BFS TOPOLOGY REPAIR (Enforce Anti-Parallel Edges)
    # Pick robust seed: largest offset (safest normal)
    # print ERROR if parallel planes detected and flipped
    if planes:
        seed = max(planes, key=lambda p: p.offset)
        queue = [seed]
        visited_idx = {seed.plane_index}
        
        while queue:
            curr = queue.pop(0)
            curr_idx = curr.plane_index
            
            # Map current directed edges
            curr_edges = {}
            n_v = len(curr.vertex_order)
            if n_v < 3: continue
            for i in range(n_v):
                u, v = curr.vertex_order[i], curr.vertex_order[(i+1)%n_v]
                curr_edges[tuple(sorted((u, v)))] = (u, v)
                
            # Check neighbors
            lookup = edge_lookup_by_plane.get(curr_idx, {})
            for key, e_idx in lookup.items():
                edge_obj = ihs_edges[e_idx]
                n_idx = edge_obj.other_plane(curr_idx)
                
                # Find neighbor object
                neighbor = next((p for p in planes if p.plane_index == n_idx), None)
                if not neighbor: continue
                
                # If unvisited, process it
                if n_idx not in visited_idx:
                    visited_idx.add(n_idx)
                    queue.append(neighbor)
                
                    # CHECK PARALLELISM
                    my_u, my_v = curr_edges.get(key, (-1,-1))
                    if my_u == -1: continue
                    
                    try:
                        n_vs = neighbor.vertex_order
                        idx_u = n_vs.index(my_u)
                        idx_v = n_vs.index(my_v)
                        # If neighbor also goes u -> v, it's PARALLEL (BAD)
                        if (idx_u + 1) % len(n_vs) == idx_v:
                            print(f"[ERROR] Parallel planes detected between planes {curr_idx} and {n_idx} - ORIENTATION LOGIC ERROR") # currently never happens; good
                            # FLIP
                            neighbor.vertex_order.reverse()
                            neighbor.vertex_inside_flags.reverse()
                            neighbor.orientation_sign *= -1
                    except ValueError:
                        pass # Vertex mismatch, disjoint topology?

    # 5. Finalize Edge Order
    for plane in planes:
        lookup = edge_lookup_by_plane.get(plane.plane_index, {})
        plane.edge_order = []
        plane.edge_vertex_pairs = []
        n_v = len(plane.vertex_order)
        if n_v >= 3:
            for i in range(n_v):
                u, v = plane.vertex_order[i], plane.vertex_order[(i+1)%n_v]
                key = (min(u, v), max(u, v))
                e_idx = lookup.get(key)
                if e_idx is not None:
                    other = ihs_edges[e_idx].other_plane(plane.plane_index)
                    plane.edge_order.append(tuple(sorted((plane.plane_index, other))))
                    plane.edge_vertex_pairs.append((u, v))
    return planes

def find_interior_point(planes: List[Plane]) -> Optional[np.ndarray]:
    """
    Finds a point x strictly inside the intersection of half-spaces defined by planes.
    Uses SciPy Highs (Linear Programming).
    
    Problem: Find x such that n_i * x < g_i for all i.
    Formulation: Maximize delta
                 Subject to: n_i * x + delta <= g_i
    
    Returns:
        x (np.ndarray): The shift vector (new origin relative to old origin).
        None: If the intersection is empty (atom fully buried).
    """
    if not planes:
        return np.zeros(3)

    # Variables: [x, y, z, delta]
    # Objective: Maximize delta => Minimize -delta
    c = [0, 0, 0, -1] 

    # Constraints: n_x*x + n_y*y + n_z*z + 1*delta <= g
    A_ub = []
    b_ub = []
    
    for p in planes:
        # n . x + delta <= g
        A_ub.append([p.n[0], p.n[1], p.n[2], 1.0])
        b_ub.append(p.g)

    # Bounds: x,y,z unbounded; delta >= 1e-5 (strictly interior)
    bounds = [(None, None), (None, None), (None, None), (1e-6, None)]

    try:
        # 'highs' is the modern, robust solver in scipy
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success and res.x[3] > 0: # delta must be positive
            return res.x[0:3]
        else:
            return None # Intersection is empty or a single point (buried)
    except Exception:
        return None

def build_ihs_for_central_atom(mol: Molecule,
                               center_idx: int,
                               neighbor_indices: Iterable[int]) -> Tuple[IHS, List[Plane]]:
    """
    Builds the IHS topology.
    Guarantee: The returned 'valid_planes_original' list is 1:1 mapped to ihs.planes indices.
    ihs.planes[i] corresponds exactly to valid_planes_original[i].
    """
    a_i = mol.atoms[center_idx]
    x_i = a_i.coord
    r_i = a_i.type.asa_radius

    # ---------------------------------------------------------
    # 1. Collect Constraints (Planes)
    # ---------------------------------------------------------
    raw_planes: List[Plane] = []
    
    # 1-A. Real Neighbors
    for nb in neighbor_indices:
        a_k = mol.atoms[nb]
        pl = circle_plane_for_neighbor(x_i, r_i, a_k.coord, a_k.type.asa_radius)
        if pl is not None:
            raw_planes.append(pl)
            
    # 1-B. Bounding Box (Virtual)
    box_size = 50.0
    box_normals = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
    for n in box_normals:
        raw_planes.append(Plane(n=np.array(n, dtype=float), g=box_size))

    if not raw_planes:
        return IHS(planes=[], vertices=[], edges=[]), []

    # ---------------------------------------------------------
    # 2. Origin Shift Logic (Prevent Singularity)
    # ---------------------------------------------------------
    min_g = min(p.g for p in raw_planes)
    x_shift = np.zeros(3)
    used_planes_for_hull = raw_planes # Default: use original

    # If origin is compromised, shift to a safe interior point
    if min_g < 1e-4:
        x_shift = find_interior_point(raw_planes)
        
        if x_shift is None:
            # Fully buried (LP failed)
            print("[OK] Atom fully buried; no accessible surface.")
            return IHS(planes=[], vertices=[], edges=[]), []

        # Shift planes: g' = g - n * x_shift
        shifted_planes = []
        for p in raw_planes:
            g_prime = p.g - np.dot(p.n, x_shift)
            shifted_planes.append(Plane(n=p.n, g=g_prime))
        used_planes_for_hull = shifted_planes

    # ---------------------------------------------------------
    # 3. Build Dual Hull & Extract Vertices
    # ---------------------------------------------------------
    dual_points = build_dual_points(used_planes_for_hull)
    
    try:
        hull = convex_hull_on_dual(dual_points)
        # Note: Vertices are in SHIFTED frame, indices refer to 'used_planes_for_hull'
        vertices_shifted = hull_faces_to_ihs_vertices(used_planes_for_hull, hull, r_i)
    except Exception: 
        return IHS(planes=[], vertices=[], edges=[]), []

    if not vertices_shifted: 
        return IHS(planes=[], vertices=[], edges=[]), []

    # ---------------------------------------------------------
    # 4. Re-Indexing & Un-Shift (The Critical Fix)
    # ---------------------------------------------------------
    
    # (4-A) Identify ACTIVE indices (referring to raw_planes)
    active_raw_indices = set()
    for v in vertices_shifted: 
        for raw_idx in v.planes: 
            active_raw_indices.add(raw_idx)

    # (4-B) Sort indices to create a deterministic mapping (0, 1, 2...)
    sorted_raw_indices = sorted(list(active_raw_indices))
    
    # Map: Raw Index (Old) -> Local Index (New, 0..N)
    raw_to_local_map = {raw: local for local, raw in enumerate(sorted_raw_indices)}
    
    # (4-C) Create the Aligned Plane List (Original Geometry)
    # This list corresponds 1:1 with the new local indices.
    valid_planes_original = [raw_planes[i] for i in sorted_raw_indices]

    final_vertices = []
    for v in vertices_shifted:
        # Restore coordinate to Original Frame
        v_coord_original = v.coord + x_shift
        
        # Check inside/outside using original radius
        dist_sq = float(np.dot(v_coord_original, v_coord_original))
        inside_flag = dist_sq < (r_i * r_i + 1e-9)

        # REMAP INDICES: Use the map to convert Raw -> Local
        new_planes = tuple(sorted(raw_to_local_map[p] for p in v.planes))
        
        final_vertices.append(IHSVertex(
            planes=new_planes, 
            coord=v_coord_original, 
            inside_sphere=inside_flag
        ))

    # ---------------------------------------------------------
    # 5. Build Topology
    # ---------------------------------------------------------
    ihs_edges = build_ihs_edges_from_face_adjacency(final_vertices)
    
    # ihs_planes will be generated with indices 0..N
    # valid_planes_original[i] provides the correct (n, g) for ihs_planes[i]
    ihs_planes = build_plane_topology(valid_planes_original, final_vertices, ihs_edges)

    return IHS(planes=ihs_planes, vertices=final_vertices, edges=ihs_edges), valid_planes_original


# ----------------------------------------------------------------
# 5) Gauss–Bonnet path: arc/vertex extraction & cycle traversal
# ----------------------------------------------------------------

class CrossingType(Enum):
    ENTER = auto()    # outside -> inside (stay on this plane after the vertex)
    EXIT = auto()     # inside -> outside (leave this plane after the vertex)
    TANGENT = auto()  # grazing contact, inside/outside unchanged

@dataclass
class GBVertex: 
    # A GB vertex is defined from IHS edges and contact with the sphere - which edge is this vertex on, and how does it cross the sphere
    # so, we store 2 things. 1) the plane that the GB arc is from 2) which plane the arc goes to
    # to achieve this, we LOOK AT BOTH PLANE EDGE DIRECTION, and check for before/after flag following the edge direction of both planes.
    # the plane that has the edge order exiting the sphere is that the GB arc goes to, and the other plane is the plane that the GB arc comes from
    # we utilize this approximate t, because for vector calculation, we will simply give 2 plane equations and find 2 vertices & arcs, so we need to know which vertex this is for the 2 points(one may not exist in GB path, just for 2 planes)
    name: str
    planes: Tuple[PlaneIndex, PlaneIndex] # this vertex is a transition between these two planes
    edge_key: EdgeKey # IHS edge this is on(2 planes)
    ihs_edge_index: int
    arc_from: PlaneIndex
    arc_to: PlaneIndex
    approximate_segment_t: float  # parameter t along the edge segment for initial placement- used to check after vector parameterization calculation
    visited: bool # utilized for traversal
    plane_crossings: Dict[PlaneIndex, CrossingType] = field(default_factory=dict)
    plane_edge_indices: Dict[PlaneIndex, int] = field(default_factory=dict)

@dataclass
class GBArc: 
    # A GB arc is defined with 1) plane it lies on 2) transition to an arc on the next plane (this corresponds to an edge in IHS) 3) transition from an arc on the previous plane
    # if, for an IHS plane which is on the sphere, if a IHS edge goes from inside the sphere to outside the sphere, assuminng we follow the edge direction of a plane k, 
    # this arc lies on plane k, so we check edge order of plane k to find if the next edge contains gb_vertex
    # note prev_plane and next_plane may be None for zero-vertice planes
    prev_plane: Optional[PlaneIndex]
    curr_plane: PlaneIndex
    next_plane: Optional[PlaneIndex]
    next_plane_edge_key: Optional[EdgeKey] # utilized for traversal
    curr_plane_edge_order_index: int # utilized for traversal - the edge index idx for the edge we are in ihs.planes[curr_plane].edge_order[idx]
    prev_gb_vertex_index: Optional[int] = None
    next_gb_vertex_index: Optional[int] = None

@dataclass
class GBLoop:
    vertex_indices: List[int]
    plane_indices: List[PlaneIndex]

#///////////////////////////////////////////////////////////////////

def _canonical_edge_key(key: EdgeKey) -> EdgeKey:
    """Return a sorted edge key for canonical hashing."""
    a, b = key
    return (a, b) if a <= b else (b, a)


class EdgeLookup:
    """
    Constant-time conversion helpers between edge indices/order slots and EdgeKeys.
    Stores both global edge index mappings and per-plane edge-order mappings.
    """

    __slots__ = (
        "_edge_key_by_index",
        "_edge_index_by_key",
        "_plane_edge_keys_by_order",
        "_plane_edge_order_idx_by_key",
    )

    def __init__(self, ihs: IHS):
        edge_keys = tuple(_canonical_edge_key(edge.planes_pair) for edge in ihs.edges)
        self._edge_key_by_index: Tuple[EdgeKey, ...] = edge_keys
        self._edge_index_by_key: Dict[EdgeKey, int] = {
            key: idx for idx, key in enumerate(edge_keys)
        }
        plane_keys: Dict[PlaneIndex, Tuple[EdgeKey, ...]] = {}
        plane_idx_lookup: Dict[PlaneIndex, Dict[EdgeKey, int]] = {}
        for plane in ihs.planes:
            oriented_keys = tuple(_canonical_edge_key(key) for key in plane.edge_order)
            plane_keys[plane.plane_index] = oriented_keys
            if not oriented_keys:
                continue
            lookup = {
                _canonical_edge_key(key): order_idx
                for order_idx, key in enumerate(oriented_keys)
            }
            plane_idx_lookup[plane.plane_index] = lookup
        self._plane_edge_keys_by_order = plane_keys
        self._plane_edge_order_idx_by_key = plane_idx_lookup

    def edge_key_from_index(self, edge_index: int) -> EdgeKey:
        return self._edge_key_by_index[edge_index]

    def edge_index_from_key(self, edge_key: EdgeKey) -> Optional[int]:
        canonical = _canonical_edge_key(edge_key)
        return self._edge_index_by_key.get(canonical)

    def plane_edge_key_from_order(self,
                                  plane_index: PlaneIndex,
                                  order_index: int) -> Optional[EdgeKey]:
        keys = self._plane_edge_keys_by_order.get(plane_index)
        if keys is None or not (0 <= order_index < len(keys)):
            return None
        return keys[order_index]

    def plane_edge_order_from_key(self,
                                  plane_index: PlaneIndex,
                                  edge_key: EdgeKey) -> Optional[int]:
        lookup = self._plane_edge_order_idx_by_key.get(plane_index)
        if lookup is None:
            return None
        return lookup.get(_canonical_edge_key(edge_key))

#///////////////////////////////////////////////////////////////////

@dataclass
class PlaneTraversalState:
    plane_index: PlaneIndex
    edge_visit_flags: List[bool]
    edge_next_vertex_cursor: List[int]

@dataclass
class GBTopology:
    vertices: List[GBVertex]
    arcs: List[GBArc]
    plane_states: Dict[PlaneIndex, PlaneTraversalState]
    zero_vertex_planes: List[PlaneIndex]
    loops: List[GBLoop]


def _segment_sphere_intersections(p0: Vec3,
                                  p1: Vec3,
                                  radius: float,
                                  tol: float = 1e-9) -> List[float]:
    """
    Return sorted parameters t in [0,1] where the segment p(t)=p0+t*(p1-p0) meets the sphere.
    """
    d = p1 - p0
    a = float(np.dot(d, d))
    if a <= tol:
        return []
    b = 2.0 * float(np.dot(p0, d))
    c = float(np.dot(p0, p0)) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < -tol:
        return []
    disc = max(disc, 0.0)
    sqrt_disc = math.sqrt(disc)
    denom = 2.0 * a
    ts = [(-b - sqrt_disc) / denom, (-b + sqrt_disc) / denom]
    unique: List[float] = []
    for t in ts:
        if -tol <= t <= 1.0 + tol: # if t is not within very nearby [0,1], we ignore it
            clamped = min(max(t, 0.0), 1.0)
            if not unique or abs(unique[-1] - clamped) > 1e-8:
                unique.append(clamped)
    unique.sort()
    return unique 


def _inside_sphere(point: Vec3, radius_sq: float, tol: float = 1e-9) -> bool: # returns True if point is inside or on the sphere
    return float(np.dot(point, point)) <= radius_sq + tol


def _crossing_from_states(before: bool, after: bool) -> CrossingType:
    if before and not after:
        return CrossingType.EXIT
    if not before and after:
        return CrossingType.ENTER
    return CrossingType.TANGENT

def extract_gb_vertices_and_arcs(x_i: np.ndarray,
                                 r_i: float,
                                 planes: List[Plane],
                                 ihs: IHS) -> GBTopology:
    """
    Extracts GB vertices using implicit crossing inversion.
    Assumes build_plane_topology has guaranteed anti-parallel shared edges.
    """
    radius_sq = r_i * r_i
    
    def get_plane_edge_def(p_idx: int, neighbor_idx: int) -> Tuple[int, int, int]:
        pl = ihs.planes[p_idx]
        target = tuple(sorted((p_idx, neighbor_idx)))
        for idx, key in enumerate(pl.edge_order):
            if key == target:
                return pl.edge_vertex_pairs[idx][0], pl.edge_vertex_pairs[idx][1], idx
        return -1, -1, -1

    gb_vertices: List[GBVertex] = []
    plane_edge_vertices_map: Dict[Tuple[int, int], List[int]] = {}

    # 1. Vertex Creation (Calculate on K, Invert for L)
    for edge in ihs.edges:
        k, l = edge.planes_pair
        
        # Get Plane K definition (Source of Truth for this edge)
        k_vs, k_ve, k_eidx = get_plane_edge_def(k, l)
        l_vs, l_ve, l_eidx = get_plane_edge_def(l, k)
        
        if k_eidx == -1 or l_eidx == -1: continue

        # Geometry on K
        pk_start = ihs.vertices[k_vs].coord
        pk_end = ihs.vertices[k_ve].coord
        vec_k = pk_end - pk_start 

        # Intersections
        ts = _segment_sphere_intersections(pk_start, pk_end, r_i)
        
        for t in ts:
            P = pk_start + t * vec_k
            v_idx = len(gb_vertices)
            
            # Crossing K
            p_before_k = P - 1e-5 * vec_k
            p_after_k  = P + 1e-5 * vec_k
            c_k = _crossing_from_states(_inside_sphere(p_before_k, radius_sq),
                                        _inside_sphere(p_after_k, radius_sq))
            
            if c_k == CrossingType.TANGENT: continue
            
            # Crossing L (IMPLICIT INVERSION)
            # Since edges are anti-parallel: Enter K == Exit L
            c_l = CrossingType.EXIT if c_k == CrossingType.ENTER else CrossingType.ENTER
            
            start_plane = -1
            if c_k == CrossingType.EXIT: start_plane = k
            elif c_l == CrossingType.EXIT: start_plane = l
            
            gb_v = GBVertex(
                name=f"GBV_{v_idx}",
                planes=(k, l),
                edge_key=edge.planes_pair,
                ihs_edge_index=-1,
                arc_from=-1,
                arc_to=start_plane,
                approximate_segment_t=t, # Relative to K
                visited=False
            )
            
            gb_v.plane_crossings[k] = c_k
            gb_v.plane_crossings[l] = c_l
            gb_v.plane_edge_indices[k] = k_eidx
            gb_v.plane_edge_indices[l] = l_eidx
            
            gb_vertices.append(gb_v)
            plane_edge_vertices_map.setdefault((k, k_eidx), []).append(v_idx)
            plane_edge_vertices_map.setdefault((l, l_eidx), []).append(v_idx)

    # 2. Sort Vertices
    plane_edge_vertices: Dict[Tuple[int, int], List[int]] = {}
    
    for key, v_indices in plane_edge_vertices_map.items():
        p_idx, e_idx = key
        if not v_indices: continue
        
        # Sort logic
        def get_sort_val(vid):
            v = gb_vertices[vid]
            # If this plane is the Reference (K), sort by t (0..1)
            # If this plane is the Neighbor (L), sort by 1-t (1..0)
            if v.planes[0] == p_idx:
                return v.approximate_segment_t
            else:
                return 1.0 - v.approximate_segment_t
                
        v_indices.sort(key=get_sort_val)
        plane_edge_vertices[key] = v_indices

    # 3. Traversal (Same as before)
    loops: List[GBLoop] = []
    zero_vertex_planes: List[PlaneIndex] = []
    
    #original code below
    for plane in ihs.planes:
        has_verts = False
        for edge_idx in range(len(plane.edge_order)):
            if plane_edge_vertices.get((plane.plane_index, edge_idx)):
                has_verts = True; break
        if not has_verts: # no other vertex with ihs pl
            if plane.vertex_order: # has IHS vertices, check if any inside sphere
                 center_chk = np.mean([ihs.vertices[v].coord for v in plane.vertex_order], axis=0) # if center point is inside sphere,
                 if _inside_sphere(center_chk, radius_sq):
                    all_inside = all(_inside_sphere(ihs.vertices[v].coord, radius_sq) for v in plane.vertex_order)
                    if not all_inside:
                        zero_vertex_planes.append(plane.plane_index)

    for start_idx, start_vertex in enumerate(gb_vertices):
        if start_vertex.visited: continue
        
        curr_plane = start_vertex.arc_to
        if curr_plane == -1: continue
        
        # Validation
        if start_vertex.plane_crossings[curr_plane] != CrossingType.EXIT:
             other = start_vertex.planes[1] if start_vertex.planes[0] == curr_plane else start_vertex.planes[0]
             if start_vertex.plane_crossings[other] == CrossingType.EXIT:
                 curr_plane = other
             else: continue

        print(f"[DEBUG] === New Loop Start at Vertex {start_idx} on Plane {curr_plane} ===")
        
        curr_vertex_idx = start_idx
        loop_vertices = []
        loop_planes = []
        loop_closed = False
        steps = 0
        
        while steps < 200:
            steps += 1
            gb_vertices[curr_vertex_idx].visited = True
            loop_vertices.append(curr_vertex_idx)
            loop_planes.append(curr_plane)
            
            curr_edge_idx = gb_vertices[curr_vertex_idx].plane_edge_indices[curr_plane]
            plane_edges = plane_edge_vertices.get((curr_plane, curr_edge_idx), [])
            
            next_vertex_idx = -1
            found = False
            
            try:
                idx = plane_edges.index(curr_vertex_idx)
                if idx + 1 < len(plane_edges):
                    next_vertex_idx = plane_edges[idx+1]
                    found = True
            except ValueError: pass
            
            if not found:
                pl_obj = ihs.planes[curr_plane]
                n_edges = len(pl_obj.edge_order)
                for i in range(1, n_edges + 1):
                    ne = (curr_edge_idx + i) % n_edges
                    cands = plane_edge_vertices.get((curr_plane, ne), [])
                    if cands:
                        next_vertex_idx = cands[0]
                        found = True
                        break
            
            if not found:
                print(f"[ERROR] Open Loop on Plane {curr_plane}")
                break
            
            next_v = gb_vertices[next_vertex_idx]
            crossing = next_v.plane_crossings[curr_plane]
            
            if crossing == CrossingType.ENTER:
                if next_vertex_idx == start_idx:
                    print(f"[DEBUG]   -> Loop Closed at V{start_idx}")
                    loop_closed = True
                    break
                p1, p2 = next_v.planes
                next_plane = p2 if p1 == curr_plane else p1
                print(f"[DEBUG]   -> V{next_vertex_idx} (ENTER). Switch {curr_plane}->{next_plane}")
                curr_plane = next_plane
                curr_vertex_idx = next_vertex_idx
            else:
                print(f"[ERROR] V{next_vertex_idx} is EXIT on Plane {curr_plane}. Logic Mismatch.")
                break

        if loop_closed:
            loops.append(GBLoop(loop_vertices, loop_planes))

    return GBTopology(gb_vertices, [], {}, zero_vertex_planes, loops)

# ======================================================================
# 6) F_in (Interior Faces) Calculation (Containment-Aware, Robust Test Point)
#    (이전 답변의 v2.0 코드를 여기에 포함합니다. chi_i 계산에 필수적입니다.)
# ======================================================================

import numpy as np
import sys
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

# Python의 재귀 깊이 한계 설정 (트리 순회용)
try:
    sys.setrecursionlimit(max(2000, sys.getrecursionlimit()))
except Exception:
    pass # Read-only
    

def is_point_inside_ihs(p: np.ndarray,
                        all_ihs_planes: List[IHSPlane],
                        tol: float = 1e-9) -> bool:
    if not np.all(np.isfinite(p)):
        return False
    for pl in all_ihs_planes:
        dot_product = np.dot(pl.unit_vec, p)
        if dot_product > pl.offset + tol:
            return False
    return True


@dataclass
class FaceNode:
    """포함 관계 트리를 위한 노드"""
    face_id: int
    is_simple_face: bool
    loop_vertices: List[np.ndarray]
    test_point: np.ndarray
    base_color_is_buried: bool   # 기본 색상 (True=Buried, False=Accessible)
    final_color_is_buried: bool = False # 최종 색상 (XOR 적용 후)
    children: List['FaceNode'] = field(default_factory=list)
    parent: Optional['FaceNode'] = None
    
    def get_root(self) -> 'FaceNode':
        """이 노드의 루트 노드를 찾습니다."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node


# ======================================================================
# 7) "Perfect" Gauss-Bonnet Area Calculation (Eq. 2, 13, 15, 16)
# ======================================================================

def _recompute_vertex_eq13(plane_k: IHSPlane,
                           plane_j: IHSPlane,
                           r_i: float,
                           tol: float = 1e-9) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    논문의 Eq. (8) ~ (13)을 구현하여 두 COI(k, j)의 교차점 2개를 계산합니다.
    P_kj = eta_kj + gamma_kj * omega_ikj
    P_jk = eta_kj - gamma_kj * omega_ikj  (논문의 Q_kl과 유사)
    """
    
    mu_k = plane_k.mu_vector # Eq. (5) [cite: 69]
    mu_j = plane_j.mu_vector
    g_k = plane_k.offset      # Eq. (6) [cite: 71, 72]
    g_j = plane_j.offset

    a_k = math.sqrt(max(0.0, r_i * r_i - g_k * g_k)) # Eq. (7)
    a_j = math.sqrt(max(0.0, r_i * r_i - g_j * g_j))
    
    # Eq. (8)
    cos_phi_kj = np.dot(mu_k, mu_j) # [cite: 119]
    sin_phi_kj_sq = 1.0 - cos_phi_kj * cos_phi_kj
    
    if sin_phi_kj_sq < tol:
        # 두 평면이 (거의) 평행함 -> 교차점 계산 불가
        print(f"[WARN] _recompute_vertex_eq13: Planes nearly parallel (sin^2(phi)={sin_phi_kj_sq})")
        return None
        
    sin_phi_kj = math.sqrt(sin_phi_kj_sq)
    
    # Eq. (10)
    tau_kj = (g_k - g_j * cos_phi_kj) / sin_phi_kj_sq # [cite: 123]
    tau_jk = (g_j - g_k * cos_phi_kj) / sin_phi_kj_sq #[cite: 123]
    
    # Eq. (9) - eta (중심점)
    eta_kj = tau_kj * mu_k + tau_jk * mu_j # [cite: 122]
    
    # Eq. (11) - omega (수직 벡터)
    omega_ikj = np.cross(mu_k, mu_j) / sin_phi_kj # [cite: 124]
    
    # Eq. (12) - gamma (반-길이)
    gamma_kj_sq = r_i*r_i - g_k*tau_kj - g_j*tau_jk # [cite: 125]
    
    if gamma_kj_sq < 0.0:
        # 두 COI가 구면 위에서 만나지 않음 (허수)
        print(f"[WARN] _recompute_vertex_eq13: No real intersection (gamma^2={gamma_kj_sq})")
        return None
        
    gamma_kj = math.sqrt(gamma_kj_sq)
    
    gamma_omega_vec = gamma_kj * omega_ikj
    
    # Eq. (13)
    P1 = eta_kj + gamma_omega_vec # [cite: 134]
    P2 = eta_kj - gamma_omega_vec # [cite: 134]
    
    return P1, P2


def _get_approximate_vertex_coord(v_gb: GBVertex, ihs: IHS) -> Optional[np.ndarray]:
    """GBVertex의 근사 좌표를 IHS 모서리로부터 계산합니다."""
    try:
        edge = ihs.edges[v_gb.ihs_edge_index]
        v_a_idx, v_b_idx = edge.vertex_ids
        v_a = ihs.vertices[v_a_idx].coord
        v_b = ihs.vertices[v_b_idx].coord

        if not np.all(np.isfinite(v_a)) or not np.all(np.isfinite(v_b)):
            return None
            
        return v_a + v_gb.approximate_segment_t * (v_b - v_a)
    except Exception:
        return None


def partition_outside_vertices(ihs, r_i: float) -> int:
    """
    [Vertex Partitioning Algorithm]
    Partitions vertices strictly outside the sphere into connected components.
    
    Rule:
      - Nodes: Vertices strictly outside the sphere.
      - Edges: IHS edges connecting these nodes that are FULLY outside (do not dip inside).
      
    Returns:
      The number of connected components in this graph, which corresponds to F_accessible.

    Time Complexity: O(V + E)
    """
    r_sq = r_i * r_i
    
    # 1. Identify Outside Vertices (Nodes) -> O(V)
    outside_node_indices = []
    for i, v in enumerate(ihs.vertices):
        # Vertex가 구 밖에 있는지만 확인
        if np.all(np.isfinite(v.coord)) and not v.inside_sphere:
            outside_node_indices.append(i)
    
    # Vertex가 하나도 없으면 F_accessible = 0 (사용자 지침: 예외처리 없이 0 반환)
    if not outside_node_indices:
        return 0
    
    outside_nodes_set = set(outside_node_indices)

    # 2. Build Adjacency List for Outside Graph -> O(E)
    adj: Dict[int, List[int]] = {i: [] for i in outside_node_indices}
    
    for edge in ihs.edges:
        v1_idx, v2_idx = edge.vertex_ids
        
        # 연결 조건 1: 양 끝점이 모두 Outside여야 함
        if (v1_idx not in outside_nodes_set) or (v2_idx not in outside_nodes_set):
            continue
            
        # 연결 조건 2: 모서리 중간이 구를 침범하면 안 됨 (Fully Outside Check)
        p1 = ihs.vertices[v1_idx].coord
        p2 = ihs.vertices[v2_idx].coord
        
        u = p2 - p1
        len_sq = np.dot(u, u)
        
        is_fully_outside = True
        if len_sq > 1e-12:
            t = -np.dot(p1, u) / len_sq # 원점에서 선분으로 내린 수선의 발 파라미터 t
            
            if 0.0 < t < 1.0:
                # 수선의 발이 선분 내부에 위치함
                closest = p1 + t * u
                # 수선의 발이 구 안쪽이면 연결 끊김 (접하는 경우는 보통 연결로 침)
                if np.dot(closest, closest) < r_sq - 1e-9: 
                    is_fully_outside = False 
        
        if is_fully_outside:
            adj[v1_idx].append(v2_idx)
            adj[v2_idx].append(v1_idx)

    # 3. Traverse (BFS) to count components -> O(V + E)
    visited: Set[int] = set()
    n_components = 0
    
    for start_node in outside_node_indices:
        if start_node in visited:
            continue
        
        n_components += 1 # 새로운 덩어리 발견
        
        # BFS 탐색으로 연결된 모든 노드 방문 처리
        queue = [start_node]
        visited.add(start_node)
        
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return n_components


def calculate_chi_user_logic(ihs, gb_topology, r_i: float) -> int:
    """
    Calculates Euler Characteristic (chi) based on User's Final Logic.
    
    Formula:
      chi = 1 - F_buried + F_accessible
    """
    
    # Step 1: F_accessible 계산 (Vertex Partitioning)
    f_accessible = partition_outside_vertices(ihs, r_i)
    
    # Step 2: N_loops 계산 (검증 없이 len 그대로 사용)
    n_loops = len(gb_topology.loops) + len(gb_topology.zero_vertex_planes)
    print("[DEBUG] Number of loops, zero-vertex planes: ", len(gb_topology.loops), len(gb_topology.zero_vertex_planes))
    # Step 3: F_buried 계산
    # 논리: 전체 면적 수(N_loops + 1)에서 F_accessible을 뺀 나머지가 F_buried
    f_buried = (n_loops + 1) - f_accessible
    
    # Step 4: Chi 계산
    # 논리: 1 - F_buried + F_accessible
    chi = 1 - f_buried + f_accessible
    
    return chi


# [calculate_asa_from_gb_topology 함수]
def calculate_asa_from_gb_topology(gb_topology: GBTopology,
                                   ihs: IHS,
                                   raw_planes: List[Plane], 
                                   r_i: float,
                                   tol: float = 1e-9) -> float:
    """
    chi_i가 (v7.0에 의해) chi_acc로 올바르게 계산되었으므로
    원본 공식 (2.0 - chi_i)를 사용합니다.
    ihs plane은 numerically instable하므로 index만 가지고 raw_planes에서 다시 참조합니다.
    """
    
    all_ihs_planes = ihs.planes 

    # --- 1. 위상 항 (Topology Term) ---
    chi_i = calculate_chi_user_logic(
        ihs, gb_topology, r_i
    )

    print(f"Debug: Calculated chi_i = {chi_i}")
    
    # [v7.0] 원본 공식 (2.0 - chi_i) 사용
    term_euler = 2.0 * math.pi * chi_i
    
    # --- 2. 기하 항 (v5.0와 동일) ---
    exact_vertex_coords: Dict[int, np.ndarray] = {}
    for v_idx, v_gb in enumerate(gb_topology.vertices):
        k, j = v_gb.planes 
        try:
            plane_k = all_ihs_planes[k]; plane_j = all_ihs_planes[j]
        except IndexError: continue
        solutions = _recompute_vertex_eq13(plane_k, plane_j, r_i)
        if solutions is None: 
            print("[ERROR] Vertex recomputation failed for GB vertex ")
            exit(-1)
            continue
        P1, P2 = solutions
        p_approx = _get_approximate_vertex_coord(v_gb, ihs)
        if p_approx is None:
            exact_vertex_coords[v_idx] = P1
            continue
        dist1_sq = np.sum((P1 - p_approx)**2); dist2_sq = np.sum((P2 - p_approx)**2)
        exact_vertex_coords[v_idx] = P1 if dist1_sq < dist2_sq else P2

    # --- 3. 경계 적분 (v5.0와 동일) ---
    term_omega_sum = 0.0
    term_phi_cos_theta_sum = 0.0

    for zero_vertex_plane_idx in gb_topology.zero_vertex_planes:
        try:
            plane = all_ihs_planes[zero_vertex_plane_idx]
        except IndexError: continue
        cos_theta = plane.offset / r_i
        if abs(cos_theta) > 1.0: 
            print("[ERROR] zero vertex cos_theta_k out of bounds")
            continue 
        term_phi_cos_theta_sum += 2.0 * math.pi * cos_theta
    for loop in gb_topology.loops:
        v_indices = loop.vertex_indices; p_indices = loop.plane_indices
        n_verts = len(v_indices)
        if n_verts < 2: 
            print ("[ERROR] loop with 0~1 vertices found")
            continue
        for i in range(n_verts):
            v_curr_idx = v_indices[i]; prev_arc_plane_idx = p_indices[i-1]
            curr_arc_plane_idx = p_indices[i]; v_next_idx = v_indices[(i+1)%n_verts]
            print(f"[DEBUG] Processing GB Arc: V{v_curr_idx} (Plane {prev_arc_plane_idx} -> {curr_arc_plane_idx}) to V{v_next_idx}")
            try:
                plane_j = all_ihs_planes[prev_arc_plane_idx]
                plane_k = all_ihs_planes[curr_arc_plane_idx]
                P_kj = exact_vertex_coords[v_curr_idx]
                P_kl_next = exact_vertex_coords[v_next_idx]
            except (KeyError, IndexError): 
                print ("[ERROR] failed to get planes or vertex coords for GB arc")
                continue
            
            g_j_sq = plane_j.offset**2; a_j_sq = r_i*r_i - g_j_sq
            if a_j_sq < tol: 
                print("[ERROR] a_j_sq TOO SMALL "+ str(a_j_sq))
                continue
            a_j = math.sqrt(a_j_sq)
            g_k_sq = plane_k.offset**2; a_k_sq = r_i*r_i - g_k_sq
            if a_k_sq < tol: 
                print("[ERROR] a_k_sq TOO SMALL "+ str(a_k_sq))
                continue
            a_k = math.sqrt(a_k_sq)

            n_j_ik = np.cross(plane_k.mu_vector, P_kj) / a_k
            m_k_ij = np.cross(plane_j.mu_vector, P_kj) / a_j
            dot_omega = max(-1.0, min(1.0, np.dot(n_j_ik, m_k_ij)))
            Omega = -math.acos(dot_omega)
            term_omega_sum += Omega

            cos_theta_k = plane_k.offset / r_i
            if abs(cos_theta_k) > 1.0: 
                print("[ERROR] cos_theta_k out of bounds")
                continue 
            m_l_ik = np.cross(plane_k.mu_vector, P_kl_next) / a_k
            dot_phi = max(-1.0, min(1.0, np.dot(n_j_ik, m_l_ik)))
            cross_phi = np.cross(n_j_ik, m_l_ik)
            S_jl_ik = math.copysign(1.0, np.dot(plane_k.mu_vector, cross_phi))
            Phi = (1.0 - S_jl_ik) * math.pi + S_jl_ik * math.acos(dot_phi)
            if Phi < 0.0:
                print("[ERROR] Negative Phi angle computed")
                continue
            if Phi > 2.0 * math.pi:
                print("[ERROR] Phi angle exceeds 2pi")
                continue
            term_phi_cos_theta_sum += Phi * cos_theta_k

    # --- 4. 최종 합산 ---
    A_i = r_i * r_i * (term_euler + term_omega_sum + term_phi_cos_theta_sum)
    if A_i < 0.0:
        print ("[WARNING] Negative ASA total area", A_i, " each term: ", term_euler, term_omega_sum, term_phi_cos_theta_sum)
    if A_i > 4.0 * math.pi * r_i * r_i:
        print ("[WARNING] ASA exceeds total sphere area by ", A_i - 4.0 * math.pi * r_i * r_i, ", each term: ", term_euler, term_omega_sum, term_phi_cos_theta_sum)

    return A_i


def asa_of_central_atom(mol: Molecule,
                        center_idx: int,
                        grid: SpatialHash) -> float:
    """
    Main driver for one atom:
      1) find candidate neighbors (grid)
      2) build IHS via dual convex hull
      3) [NEW] Pre-check for Fully Accessible / Fully Buried
      4) extract GB path (vertices + arcs)
      5) compute arc lengths, exterior angles, and sum Gauss–Bonnet terms
      6) return ASA_i
    """
    a_i = mol.atoms[center_idx]
    x_i = a_i.coord
    r_i = a_i.type.asa_radius
    radius_sq = r_i * r_i

    cand = list(grid.neighbors(mol, center_idx))

    ihs, plane_records = build_ihs_for_central_atom(mol, center_idx, cand)
    
    # --- [DEBUG] IHS Statistics ---
    print(f"[DEBUG] Atom {center_idx}: IHS Built -> "
          f"Planes={len(ihs.planes)}, "
          f"Vertices={len(ihs.vertices)}, "
          f"Edges={len(ihs.edges)}")
    # ------------------------------
    
    # === [Pre-check 1] ===
    if not ihs.vertices: # 0.0 by logic(don't calculate anything further)
        return 0.00

    gb_topology = extract_gb_vertices_and_arcs(x_i, r_i, plane_records, ihs)
    
    if not gb_topology.loops and not gb_topology.arcs:
        # Specialized check: intersection topology exists but no valid GB loops found.
        # Check if IHS vertices are inside the sphere (buried) or outside (exposed).
        all_vertices_inside = all(v.inside_sphere for v in ihs.vertices)
        
        if all_vertices_inside:
            print("every point is inside the IHS, fully buried case")
            return 0.0
        else:
            print("Warning?: every point is outside the IHS, fully exposed case - this is a BUG if there are any more than 8 vertices")
            return 4.0 * math.pi * radius_sq

    # === [Partially Buried Case] ===
    final_asa = calculate_asa_from_gb_topology(
        gb_topology,
        ihs,
        plane_records, 
        r_i
    )
    
    return final_asa


# --------------------------------
# 7) Whole-molecule ASA (skeleton)
# --------------------------------

def compute_asa(mol: Molecule,
                cell_size: float = 2.0 * (2.0 + PROBE_RADIUS)) -> List[float]:
    """
    Compute ASA for all atoms. Default cell size: twice max vdW (≈ 2.0 Å here) + probe.
    """
    grid = SpatialHash(cell_size=cell_size)
    grid.build(mol)

    asas: List[float] = []
    for i in range(len(mol.atoms)):
        ai = asa_of_central_atom(mol, i, grid)
        asas.append(ai)
    return asas


# ------------------------
# 8) CLI entry (quick run)
# ------------------------

# ------------------------
# 8) CLI entry (Modified)
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Analytical ASA via Gauss–Bonnet/IHS.")
    ap.add_argument("pdb", help="PDB file path")
    args = ap.parse_args()

    # Extract PDB ID from filename (e.g., "data/1ubq.pdb" -> "1ubq")
    pdb_id = os.path.splitext(os.path.basename(args.pdb))[0]
    csv_filename = f"{pdb_id}.csv"

    try:
        mol = parse_pdb(args.pdb)
    except Exception as e:
        print(f"Error parsing PDB: {e}")
        sys.exit(1)

    print(f"Computing ASA for {len(mol.atoms)} atoms...")
    asas = compute_asa(mol)

    # --- Statistics Aggregation ---
    polar_area = 0.0
    apolar_area = 0.0
    unknown_area = 0.0
    
    surface_cnt = 0
    buried_cnt = 0
    
    # Tolerance for float comparison to zero
    TOLERANCE = 1e-6

    # List to hold data for CSV writing
    csv_rows = []

    for atom, area in zip(mol.atoms, asas):
        # 1. Count Surface vs Buried
        if area > 0.0 :
            surface_cnt += 1
        else:
            buried_cnt += 1
        
        # 2. Sum Polar/Apolar/Unknown Area
        # using the AtomClass enum logic defined in your code
        if atom.type.klass == AtomClass.OTHER:
            unknown_area += area
        elif atom.type.polar:
            polar_area += area
        else:
            apolar_area += area

        # 3. Prepare row for CSV
        # Columns: ATOM_ID, ATOM_NAME, RES_NAME, RES_SEQ, AREA
        csv_rows.append([
            atom.idx + 1,
            atom.name,
            atom.res_name,
            atom.res_seq,
            f"{area:.4f}"
        ])

    total_area = sum(asas)

    # --- Output 1: Write to CSV ---
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ATOM", "NAME", "RESIDUE", "SEQ", "AREA"])
            writer.writerows(csv_rows)
        print(f"\nSuccessfully saved per-atom data to: {csv_filename}\n")
    except IOError as e:
        print(f"\nError writing CSV file: {e}\n")

    # --- Output 2: Print Summary (Matching Image Format) ---
    # Using specific spacing to match the screenshot provided
    
    print(f"{'POLAR  area/energy':<25} = {polar_area:>12.2f}")
    print(f"{'APOLAR area/energy':<25} = {apolar_area:>12.2f}")
    print(f"{'UNKNOWN area/energy':<25} = {unknown_area:>12.2f}")
    
    print("-" * 42)
    print(f"{'Total  area/energy':<25} = {total_area:>12.2f}")
    print("-" * 42)
    print("")
    
    print(f"{'Number of surface atoms':<25} = {surface_cnt:>12}")
    print(f"{'Number of buried atoms':<25} = {buried_cnt:>12}")

if __name__ == "__main__":
    main()
