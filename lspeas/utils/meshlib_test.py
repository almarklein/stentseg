import time

import numpy as np
from pytest import raises

import meshlib


def test_maker_funcs():
    for func in [meshlib.make_cube, meshlib.make_tetrahedron, meshlib.make_icosahedron, meshlib.make_sphere]:

        vertices = func()

        # The make functions return simple flat vertices with implicit faces
        assert isinstance(vertices, np.ndarray)
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
    vertices[3:-3] = vertices[6:]
    vertices = vertices[:-3]
    # This is detected in early validation
    with raises(ValueError):
        m = meshlib.Mesh(vertices)

    # Get vertices remove one face
    vertices = meshlib.make_icosahedron()
    vertices[3:-3] = vertices[6:]
    vertices = vertices[:-3]
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


def test_plane_cut():

    # On a tetrahedron
    m1 = meshlib.Mesh(meshlib.make_tetrahedron())
    assert m1._vertices.shape[0] == 4
    assert m1._faces.shape[0] == 4
    #
    m2 = m1.cut_plane((0, 0, +1, 0))
    assert m2.ensure_closed() == 0
    assert m2._vertices.shape[0] == 4 + 3 + 1
    assert m2._faces.shape[0] == 4 - 1 + 3
    #
    m2 = m1.cut_plane((0, 0, -1, 0))
    assert m2.ensure_closed() == 0
    assert m2._vertices.shape[0] == 4 + 3 + 1
    assert m2._faces.shape[0] == 4 + 3 + 3

    # On a cube
    m1 = meshlib.Mesh(meshlib.make_cube())
    assert np.isclose(m1.volume(), 8)
    assert m1._vertices.shape[0] == 8
    assert m1._faces.shape[0] == 12
    #
    m2 = m1.cut_plane((0, 0, 1, 0))
    assert m2.ensure_closed() == 0
    assert np.isclose(m2.volume(), 4)
    assert m2._vertices.shape[0] == 8 + 8 + 1
    assert m2._faces.shape[0] == 12 - 2 + 8 + 4

    # On a sphere
    m1 = meshlib.Mesh(meshlib.make_sphere())
    assert 4 < m1.volume() < 4.2  # 4.188790 == pi * 4/3
    #
    m2 = m1.cut_plane((0, 0, 1, 0))
    assert m2.ensure_closed() == 0
    assert np.isclose(m2.volume(), m1.volume() / 2)
    assert m2._vertices.shape[0] > m1._vertices.shape[0]
    assert m2._faces.shape[0] < m1._faces.shape[0]  # the lid takes less than what we discard

    # Test on more spheres (tests the grouping over multiple rims)
    vertices = np.row_stack([meshlib.make_sphere() + np.array([5, 0, 0]),
                             meshlib.make_sphere() + np.array([10, 0, 0]),
                             meshlib.make_sphere() + np.array([15, 0, 0]),
                             ])
    m1 = meshlib.Mesh(vertices)
    assert 12 < m1.volume() < 12.6
    #
    m2 = m1.cut_plane((0, 0, 1, 0))
    assert m2.ensure_closed() == 0
    assert np.isclose(m2.volume(), m1.volume() / 2)
    assert m2._vertices.shape[0] > m1._vertices.shape[0]
    assert m2._faces.shape[0] < m1._faces.shape[0]

    # Check that the correct side is removed - part 1
    m1 = meshlib.Mesh(meshlib.make_tetrahedron())
    m2 = m1.cut_plane((0, 0, 1, 0))  # should keep upper part
    assert m2.volume() < m1.volume() / 2
    #
    m2 = m1.cut_plane((0, 0, -1, 0))  # should keep lower part
    assert m2.volume() > m1.volume() / 2

    # Check that the correct side is removed - part 2
    m1 = meshlib.Mesh(meshlib.make_sphere())
    m2 = m1.cut_plane((0, 0, 1, 0))  # middle,  keep upper part
    assert m2.get_flat_vertices()[:, 2].min() == 0
    assert m2.get_flat_vertices()[:, 2].max() == 1
    #
    m2 = m1.cut_plane((0, 0, -1, 0))  # middle,  keep bottom part
    assert m2.get_flat_vertices()[:, 2].min() == -1
    assert m2.get_flat_vertices()[:, 2].max() == 0
    #
    m2 = m1.cut_plane((0, 0, 1, -0.5))  # should keep small lower part
    assert np.isclose(m2.get_flat_vertices()[:, 2].min(), 0.5)
    assert m2.get_flat_vertices()[:, 2].max() == 1
    assert m2.volume() < m1.volume() / 4
    #
    m2 = m1.cut_plane((0, 0, -1, 0.5))  # should keep big lower part
    assert m2.get_flat_vertices()[:, 2].min() == -1
    assert np.isclose(m2.get_flat_vertices()[:, 2].max(), 0.5)
    assert m2.volume() > 3 * m1.volume() / 4

    # Test what happens if we cut near a vertex
    m1 = meshlib.Mesh(meshlib.make_tetrahedron())
    assert m2.ensure_closed() == 0
    assert m1._faces.shape[0] == 4
    #
    m2 = m1.cut_plane((0, 0, 1, -1.01))  # cut just above top vertex
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 0
    #
    m2 = m1.cut_plane((0, 0, 1, +1))  # cut below lowest vertex
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 4
    #
    m2 = m1.cut_plane((0, 0, 1, -0.99))  # cut just below top vertex
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 6
    assert m2.volume() < 0.01 * m1.volume()
    #
    m2 = m1.cut_plane((0, 0, -1, 0.99))  # cut just below top vertex - flipped
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 10
    assert m2.volume() > 0.99 * m1.volume()
    #
    m2 = m1.cut_plane((0, 0, 1, -1))  # cut at top vertex
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 6  # not 0, 4 would have been better
    assert m2.volume() == 0
    #
    m2 = m1.cut_plane((0, 0, -1, 1))  # cut at top vertex - flipped
    assert m2.ensure_closed() == 0
    assert m2._faces.shape[0] == 4  # untouched


def speed():
    vertices = meshlib.make_sphere(5)
    t0 = time.perf_counter()
    m = meshlib.Mesh(vertices)
    t1 = time.perf_counter()
    m.ensure_closed()
    t2 = time.perf_counter()
    m.volume()
    t3 = time.perf_counter()
    print("Sphere", len(vertices), t1 - t0, t2 - t1, t3 - t2)


if __name__ == "__main__":
    test_maker_funcs()
    test_fix_wrong_winding()
    test_detect_holes()
    test_plane_cut()

