import numpy as np
from stentseg.utils import PointSet

import meshlib
from pytest import raises


def test_maker_funcs():
    for func in [meshlib.make_cube, meshlib.make_tetrahedron, meshlib.make_icosahedron, meshlib.make_sphere]:

        vertices = func()

        # The make functions return simple flat vertices with implicit faces
        assert isinstance(vertices, PointSet)
        assert vertices.ndim == 2 and vertices.shape[1] == 3

        # The mesh class can convert them and the meshes are closed
        m = meshlib.Mesh(vertices)
        m_vertices, m_faces = m.get_vertices_and_faces()
        assert len(m_vertices) < len(vertices)
        assert len(m_faces)
        assert m.ensure_closed() == 0

        # The meshes are all on/within -1, 1 in all dimensions
        assert m_vertices.min() >= -1
        assert m_vertices.max() <= 1
        assert m.volume() <= 8
        assert m.volume() >= 0.5

        # Making it flat should produce original data
        assert np.all(vertices == m.get_flat_vertices())

    # Extra check for sphere
    for n in range(4):
        m = meshlib.Mesh(meshlib.make_sphere(1))
        assert m.ensure_closed() == 0


def test_fix_wrong_winding():

    # Get vertices and break one
    vertices = meshlib.make_icosahedron() + 5
    vertices[2], vertices[1] = vertices[1].copy(), vertices[2].copy()

    # Auto unwind!
    m = meshlib.Mesh(vertices)
    assert m.ensure_closed() == 1
    m.ensure_closed() == 0


def test_detect_holes():

    # Get vertices remove one face
    vertices = meshlib.make_tetrahedron()
    vertices.pop(3)
    vertices.pop(3)
    vertices.pop(3)
    # This is detected in early validation
    with raises(ValueError):
        m = meshlib.Mesh(vertices)

    # Get vertices remove one face
    vertices = meshlib.make_icosahedron()
    vertices.pop(3)
    vertices.pop(3)
    vertices.pop(3)
    # This is detected when ensuring it is closed
    m = meshlib.Mesh(vertices)
    with raises(ValueError):
        m.ensure_closed()

    # Make a good mesh
    vertices = meshlib.make_icosahedron()
    m = meshlib.Mesh(vertices)
    assert m.ensure_closed() == 0
    # then change in-place
    m._faces[1] = m._faces[2]
    with raises(ValueError):
        m.ensure_closed()


if __name__ == "__main__":
    test_maker_funcs()
    test_fix_wrong_winding()
    test_detect_holes()

