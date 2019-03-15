import numpy as np


class Mesh:
    """ A class to represent a geometric mesh.

    Can be instantiated as:

    * Mesh(vertices, faces)
    * Mesh(vertices)  # the implicit will be automatically detected

    Winding: this class adopts the right hand-rule to determine what
    is inside or out. This is similar to STL and e.g. Blender (but
    different from Unity). It means that the winding is in the direction
    of the fingers of your right hand (i.e. counter clockwise) and the
    normal (i.e. outside) will be in the direction of your thumb.
    """

    def __init__(self, vertices, faces=None, v2f=None):
        if faces is None:
            self._from_vertices(vertices)
        else:
            self._vertices = np.asarray(vertices, dtype=np.float32)
            self._faces = np.asarray(faces, dtype=np.int32)
            if v2f is None:
                self._v2f = self._calculate_v2f(self._faces)
            else:
                self._v2f = v2f

        self.validate()

    def __repr__(self):
        t = "<Mesh with {} vertices and {} faces at 0x{}>"
        return t.format(len(self._vertices), len(self._faces), hex(id(self)).upper())

    def get_vertices_and_faces(self):
        """ Get the vertices and faces arrays.
        These are the original (internal) arrays, don't edit!
        """
        return self._vertices, self._faces

    def get_flat_vertices(self):
        """ Get a representation of the mesh as flat vertices, e.g. to
        export to STL.
        """
        faces = self._faces
        vertices = self._vertices
        flat_vertices = np.zeros((0, 3), np.float32)
        for fi in range(len(faces)):
            vi1, vi2, vi3 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
            append3(flat_vertices, vertices[vi1])
            append3(flat_vertices, vertices[vi2])
            append3(flat_vertices, vertices[vi3])
        return flat_vertices

    def _calculate_v2f(self, faces):
        """ Calculate the v2f map from the given faces.
        """
        v2f = {}
        for fi in range(len(faces)):
            vi1, vi2, vi3 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
            v2f.setdefault(vi1, []).append(fi)
            v2f.setdefault(vi2, []).append(fi)
            v2f.setdefault(vi3, []).append(fi)
        return v2f

    def _from_vertices(self, vertices_in):
        """ Create a mesh from only the vertices (e.g. from STL) by
        recombining equal vertices into faces.
        """
        self._vertices = vertices = np.zeros((0, 3), np.float32)
        self._faces = faces =  np.zeros((0, 3), np.int32)
        self._v2f = v2f = {}

        xyz2f = {}

        if not(vertices_in.ndim == 2 and vertices_in.shape[1] == 3):
            raise ValueError("Vertices must be an Nx3 array.")
        if len(vertices_in) % 3 != 0:
            raise ValueError("There must be a multiple of 3 vertices.")

        for fi in range(len(vertices_in) // 3):
            fi2 = len(faces)
            face = []
            for vi in (fi * 3 + 0, fi * 3 + 1, fi * 3 + 2):
                xyz = vertices_in[vi]
                xyz = float(xyz[0]), float(xyz[1]), float(xyz[2])
                if xyz in xyz2f:
                    vi2 = xyz2f[xyz]  # re-use vertex
                else:
                    vi2 = len(vertices)  # new vertex
                    append3(vertices, xyz)
                    xyz2f[xyz] = vi2
                face.append(vi2)
                faceslist = v2f.setdefault(vi2, [])
                faceslist.append(fi2)
            append3(faces, face)

    def validate(self):
        """ perform basic validation on the mesh.
        """
        assert isinstance(self._vertices, np.ndarray)
        assert self._vertices.ndim == 2 and self._vertices.shape[1] == 3

        assert isinstance(self._faces, np.ndarray)
        assert self._faces.ndim == 2 and self._faces.shape[1] == 3

        assert isinstance(self._v2f, dict)

        # The vertices in faces all exist
        vertices_in_faces = set()
        all_vertices = set(range(len(self._vertices)))
        for fi in range(len(self._faces)):
            for i in range(3):
                vertices_in_faces.add(self._faces[fi, i])
        if vertices_in_faces.difference(all_vertices):
            raise ValueError("Some faces refer to nonexisting vertices")
        # NOTE: there may still be unused vertices!

        # All vertices are in at least 3 faces. This is a basic sanity check
        # but does not mean that the mesh is closed!
        counts = set()
        for vi, faces in self._v2f.items():
            counts.add(len(faces))
        for count in counts:
            if count < 3:
                raise ValueError("was expecting all vertices to be in at least 3 faces.")

    def ensure_closed(self):
        """ Ensurs that the mesh is closed, that all faces have the
        same winding ,and that the winding follows the right hand rule
        (by checking that the volume is positive). It is recommended
        to call this on incoming data.

        Returns the number of faces that were changed to correct winding.
        """
        vertices = self._vertices
        faces = self._faces
        v2f = self._v2f

        faces_to_check = set(range(len(faces)))
        count_reversed = 0

        while faces_to_check:
            front = set([faces_to_check.pop()])
            while front:
                fi_check = front.pop()
                vi1, vi2, vi3 = faces[fi_check, 0], faces[fi_check, 1], faces[fi_check, 2]
                edgesthathaveneighbours = set()
                neighbour_faces = set()
                neighbour_faces.update(v2f[vi1])
                neighbour_faces.update(v2f[vi2])
                neighbour_faces.update(v2f[vi3])
                for fi in neighbour_faces:
                    vj1, vj2, vj3 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
                    matching_vertices = {vj1, vj2, vj3}.intersection({vi1, vi2, vi3})
                    if len(matching_vertices) >= 2:
                        if {vi1, vi2} == matching_vertices:
                            edgesthathaveneighbours.add(1)
                            if fi in faces_to_check:
                                faces_to_check.discard(fi)
                                front.add(fi)
                                if ((vi1 == vj1 and vi2 == vj2) or
                                    (vi1 == vj2 and vi2 == vj3) or
                                    (vi1 == vj3 and vi2 == vj1)):
                                    count_reversed += 1
                                    faces[fi, 1], faces[fi, 2] = int(faces[fi, 2]), int(faces[fi, 1])
                        elif {vi2, vi3} == matching_vertices:
                            edgesthathaveneighbours.add(2)
                            if fi in faces_to_check:
                                faces_to_check.discard(fi)
                                front.add(fi)
                                if ((vi2 == vj1 and vi3 == vj2) or
                                    (vi2 == vj2 and vi3 == vj3) or
                                    (vi2 == vj3 and vi3 == vj1)):
                                    count_reversed += 1
                                    faces[fi, 1], faces[fi, 2] = int(faces[fi, 2]), int(faces[fi, 1])
                        elif {vi3, vi1} == matching_vertices:
                            edgesthathaveneighbours.add(3)
                            if fi in faces_to_check:
                                faces_to_check.discard(fi)
                                front.add(fi)
                                if ((vi3 == vj1 and vi1 == vj2) or
                                    (vi3 == vj2 and vi1 == vj3) or
                                    (vi3 == vj3 and vi1 == vj1)):
                                    count_reversed += 1
                                    faces[fi, 1], faces[fi, 2] = int(faces[fi, 2]), int(faces[fi, 1])
                # Now that we checked all neighbours, check if we have a neighbour on each edge.
                # If this is so for all faces, we know that the mesh is closed. The mesh
                # can still have weird crossovers or parts sticking out (e.g. a Klein bottle).
                if edgesthathaveneighbours != {1, 2, 3}:
                    msg = "there is a hole in the mesh at face {} {}".format(fi_check, edgesthathaveneighbours)
                    #print("WARNING:", msg)
                    raise ValueError(msg)

        # todo: this check should really be done to each connected component within the mesh.
        # For now we assume that the mesh is on single object.

        # Reverse all?
        if self.volume() < 0:
            faces[:,1], faces[:,2] = faces[:,2].copy(), faces[:,1].copy()
            count_reversed = len(faces) - count_reversed

        return count_reversed

    def volume(self):
        """ Calculate the volume of the mesh. You probably want to run ensure_closed()
        on untrusted data when using this.
        """

        # https://stackoverflow.com/a/1568551
        def _volume_of_triangle(p1, p2, p3):
            v321 = p3[0] * p2[1] * p1[2]
            v231 = p2[0] * p3[1] * p1[2]
            v312 = p3[0] * p1[1] * p2[2]
            v132 = p1[0] * p3[1] * p2[2]
            v213 = p2[0] * p1[1] * p3[2]
            v123 = p1[0] * p2[1] * p3[2]
            return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)

        vertices = self._vertices
        faces = self._faces

        vol1 = 0
        # vol2 = 0
        for fi in range(len(faces)):
            vi1, vi2, vi3 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
            vol1 += _volume_of_triangle(vertices[vi1], vertices[vi2], vertices[vi3])
            # vol2 += _volume_of_triangle(vertices[vi1] + 10, vertices[vi2] + 10, vertices[vi3] + 10)

        # # Check integrity
        # err_per = (abs(vol1) - abs(vol2)) / max(abs(vol1) + abs(vol2), 0.000000001)
        # if err_per > 0.1:
        #     raise RuntimeError("Cannot calculate volume, the mesh looks not to be closed!")
        # elif err_per > 0.001:
        #     print("WARNING: maybe the mesh is not completely closed?")

        return vol1

    def cut_plane(self, plane):
        """ Cut the part of the mesh that is in front of the given plane.
        """

        # This mesh
        faces = self._faces
        vertices = self._vertices

        # Prepare for creating a new mesh
        new_faces = np.zeros((0, 3), np.int32)  # Build up from scratch
        new_vertices = self._vertices.copy()  # Start with full, decimate later
        edge_to_vertex_index = {}

        # Get signed distance of each vertex to the plane
        signed_distances = signed_distance_to_plane(vertices, plane)

        def get_new_vertex_id(vi1, vi2):
            key = min(vi1, vi2), max(vi1, vi2)
            try:
                return edge_to_vertex_index[key]
            except KeyError:
                d1, d2 = abs(signed_distances[vi1]), abs(signed_distances[vi2])
                w1, w2 = d2 / (d1 + d2), d1 / (d1 + d2)
                vertex = w1 * vertices[vi1] + w2 * vertices[vi2]
                new_vi = len(new_vertices)
                new_vertices.append(vertex)
                edge_to_vertex_index[key] = new_vi
                return new_vi

        # Now iterate over the faces (triangles), and check each edge. If the two
        # points are on different sides of the plane, then we interpolate on the
        # edge to get the exact spot where the edge intersects the plane.
        for fi in range(len(faces)):  # for each face index
            vi1, vi2, vi3 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
            s1, s2, s3 = signed_distances[vi1], signed_distances[vi2], signed_distances[vi3]
            include = True
            if s1 >= 0 and s2 >= 0 and s3 >= 0:
                pass  # The whole face is to be dropped
            elif s1 < 0 and s2 < 0 and s3 < 0:
                # The whole face is to be included
                new_faces.append(faces[fi])
            elif s1 < 0 and s2 >= 0 and s3 >= 0:
                a = vi1
                b = get_new_vertex_id(vi1, vi2)
                c = get_new_vertex_id(vi1, vi3)
                new_faces.append(a, b, c)
            elif s2 < 0 and s3 >= 0 and s1 >= 0:
                a = vi2
                b = get_new_vertex_id(vi2, vi3)
                c = get_new_vertex_id(vi2, vi1)
                new_faces.append(a, b, c)
            elif s3 < 0 and s1 >= 0 and s2 >= 0:
                a = vi3
                b = get_new_vertex_id(vi3, vi1)
                c = get_new_vertex_id(vi3, vi2)
                new_faces.append(a, b, c)
            elif s1 < 0 and s2 < 0 and s3 >= 0:
                a = vi1
                b = vi2
                c = get_new_vertex_id(vi1, vi3)
                d = get_new_vertex_id(vi2, vi3)
                new_faces.append(a, b, d)
                new_faces.append(b, c, d)
            elif s2 < 0 and s3 < 0 and s1 >= 0:
                a = vi2
                b = vi3
                c = get_new_vertex_id(vi2, vi1)
                d = get_new_vertex_id(vi3, vi1)
                new_faces.append(a, b, d)
                new_faces.append(b, c, d)
            elif s3 < 0 and s1 < 1 and s2 >= 0:
                a = vi3
                b = vi1
                c = get_new_vertex_id(vi3, vi2)
                d = get_new_vertex_id(vi1, vi2)
                new_faces.append(a, b, d)
                new_faces.append(b, c, d)
            else:
                assert False, "Unforeseen, this should not happen"

        # Create v2f map
        new_v2f = self._calculate_v2f(new_faces)

        # Find the different holes that make up the rim of the mesh
        groups = []
        rim_indices_left = set(range(len(self._vertices), len(new_vertices)))
        rim_indices_done = set()
        while rim_indices_left:
            # Pick arbitrarty vertex on the rim. This will be our seed for a new group.
            vi = rim_indices_left.pop()
            group = [vi]
            groups.append(group)
            # Now walk along the rim until we're back
            while not (len(group) >= 2 and group[0] == group[-1]):
                faces = new_v2f[vi]
                vi_next = None
                for fi in faces:
                    # If the vertex next to the current vertex (in the correct direction/winding)
                    # is on the rim, make it the next current.
                    vi1, vi2, vi3 = new_faces[fi, 0], new_faces[fi, 1], new_faces[fi, 2]
                    if vi1 == vi and vi2 in rim_indices_left:
                        vi_next = vi2
                        break
                    elif vi2 == vi and vi3 in rim_indices_left:
                        vi_next = vi3
                        break
                    elif vi3 == vi and vi1 in rim_indices_left:
                        vi_next = vi1
                        break
                else:
                    continue
                break
                # Done, or next
                if vi_next is None:
                    raise RuntimeError("Could not find next vertex on the rim")
                vi = vi_next
                rim_indices_left.remove(vi)
                group.append(vi)
                continue

        # Put the lid on each hole. Each group is ordered with correct winding already.
        for group in groups:
            center_vertex = sum(new_vertices[vi] for vi in group) / len(group)
            center_vi = len(vertices_new)
            vertices_new.add(center_vertex)
            new_v2f[center_vi] = fis = []
            for i in range(len(group) -1):
                fis.append(len(new_faces))
                new_faces.add(group[i], group[i + 1], center_vi)
            fis.append(len(new_faces))
            new_faces.add(group[-1], group[0], center_vi)

        return Mesh(new_vertices, new_faces, new_v2f)


## Util functions


def append3(arr, p):
    arr.resize((arr.shape[0] + 1, arr.shape[1]), refcheck=False)
    arr[-1] = p


def norm(p):
    return (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5


def signed_distance_to_plane(pp, plane):
    a, b, c, d = plane
    plane_norm = (a**2 + b**2 + c**2) ** 0.5
    return (a * pp[:, 0] + b * pp[:, 1] + c * pp[:, 2] + d) / plane_norm


## Maker functions


def make_cube():
    """ Create a vertex array representing a cube centered at the origin,
    spanning 1 unit in each direction (thus having a volume of 8).
    """
    vertices =  np.zeros((0, 3), np.float32)
    for rot in [0, 1, 2]:
        for c in [-1, +1]:
            a1, a2 = -1 * c, +1 * c
            b1, b2 = -1, +1
            for values in [(a1, b1, c), (a2, b2, c), (a1, b2, c),
                           (a1, b1, c), (a2, b1, c), (a2, b2, c)]:
                values = values[rot:] + values[:rot]
                append3(vertices, values)
    return vertices


def make_tetrahedron():
    """ Create a vertex array representing a tetrahedron (pyramid)
    centered at the origin, with its vertices on the unit sphere. The
    tatrahedon is the 3D object with the least possible number of faces.
    """
    sqrt = lambda x: x**0.5
    # Points on unit sphere
    v1 = sqrt(8/9), 0, -1/3
    v2 = -sqrt(2/9), sqrt(2/3), -1/3
    v3 = -sqrt(2/9), -sqrt(2/3), -1/3
    v4 = 0, 0, 1
    # Create faces
    vertices = np.zeros((0, 3), np.float32)
    for v1, v2, v3 in [(v1, v2, v4), (v2, v3, v4), (v3, v1, v4), (v1, v3, v2)]:
        append3(vertices, v1)
        append3(vertices, v2)
        append3(vertices, v3)
    return vertices


def make_icosahedron():
    """ Create a vertex array representing an icosahedron (a polyhedron
    with 20 faces) centered at the origin, with its vertices on the
    unit sphere.
    """
    # Inspired from the Red book, end of chaper 2.

    X = 0.525731112119133606
    Z = 0.850650808352039932

    vdata = [
        (-X, 0.0, Z), (X, 0.0, Z), (-X, 0.0, -Z), (X, 0.0, -Z),
        (0.0, Z, X), (0.0, Z, -X), (0.0, -Z, X), (0.0, -Z, -X),
        (Z, X, 0.0), (-Z, X, 0.0), (Z, -X, 0.0), (-Z, -X, 0.0),
    ]

    faces = [
        (0,4,1), (0,9,4), (9,5,4), (4,5,8), (4,8,1),
        (8,10,1), (8,3,10), (5,3,8), (5,2,3), (2,7,3),
        (7,10,3), (7,6,10), (7,11,6), (11,0,6), (0,1,6),
        (6,1,10), (9,0,11), (9,11,2), (9,2,5), (7,2,11)
    ]

    vertices = np.zeros((0, 3), np.float32)
    for v1, v2, v3 in faces:
        append3(vertices, vdata[v1])
        append3(vertices, vdata[v3])  # note the out-of order to make CCW winding
        append3(vertices, vdata[v2])

    return vertices


def make_sphere(ndiv=3):
    """ Create a vertex array representing a unit sphere centered at
    the origin. The vertices are generated by subdividing an icosahedron.
    """
    vertices = make_icosahedron()
    for iter in range(ndiv):
        new_vertices = np.zeros((0, 3), np.float32)
        for vi0 in range(0, len(vertices), 3):
            v1, v2, v3 = vertices[vi0 + 0], vertices[vi0 + 1], vertices[vi0 + 2]
            v4, v5, v6 = 0.5 * (v1 + v2), 0.5 * (v2 + v3), 0.5 * (v3 + v1)
            v4, v5, v6 = v4 / norm(v4), v5 / norm(v5), v6 / norm(v6)
            for vi in [v1, v4, v6, v2, v5, v4, v3, v6, v5, v4, v5, v6]:
                append3(new_vertices, vi)
        vertices = new_vertices
    return vertices



if __name__ == "__main__":
    import os
    import visvis as vv
    from stentseg.utils.datahandling import loadmesh

    ptcode = 'LSPEAS_003'
    ctcode1 = 'discharge'
    basedirMesh = r"C:\stack\data\lspeas\vaatwand"

    # filename ='{}_{}_neck.stl'.format(ptcode, ctcode1)
    # vessel1 = loadmesh(basedirMesh, ptcode[-3:], filename) #inverts Z
    # vv.processing.unwindFaces(vessel1)
    # m = from_vertices(vessel1._vertices)

    plane = (0.23038404068509294, -0.2301466100921624, 0.945492322370042, -102.36959192746241)
    # m.cut_plane(plane)


    # Make a sphere with one whole in it
    s = Mesh(make_sphere())  # should have volume of 4/3 * np.pi * 0.5**3 = 0.5236
    q = Mesh(make_cube())  # should have volume of 1
    # s._faces[3,1], s._faces[3,2] = s._faces[3,2], s._faces[3,1]
