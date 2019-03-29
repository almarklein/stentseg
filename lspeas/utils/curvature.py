"""
Measure curvature.
"""

import numpy as np


def measure_curvature(pp, deforms):
    """ Take curvature measurements. With 10 phases:
    * mean curvature per phase (list of 10 values)
    * max curvature per phase (list of 10 values)
    * max_curvature locations list of 10 positions)
    * tuple (position, max-value) for the point with the most curvature change.
    """
    pp = np.asarray(pp)

    # Calculate curvature array in each phase
    curvatures_per_phase = []
    for phase in range(len(deforms)):
        deform = deforms[phase]
        dx = deform.get_field_in_points(pp, 0)
        dy = deform.get_field_in_points(pp, 1)
        dz = deform.get_field_in_points(pp, 2)
        deform_vectors = np.stack([dx, dy, dz], 1)
        curvatures_per_phase.append(get_curvatures(pp + deform_vectors))

    # Mean curvature per phase (1 value per phase)
    mean_per_phase = []
    for curvatures in curvatures_per_phase:
        mean_per_phase.append(float(curvatures.mean()))

    # Max curvature per phase and position (1 tuple per phase)
    max_per_phase = []
    max_per_phase_loc = []
    for curvatures in curvatures_per_phase:
        index = np.argmax(curvatures)
        max_per_phase.append((float(curvatures[index])))
        max_per_phase_loc.append(length_along_path(pp, index))

    # Max change (index, max-value)
    max_index, max_change, max_value = 0, 0, 0
    for index in range(len(pp)):
        curvature_per_phase = [float(curvatures_per_phase[phase][index]) for phase in range(len(deforms))]
        change = max(curvature_per_phase) / min(curvature_per_phase)
        if change > max_change:
            max_index, max_change, max_value = index, change, max(curvature_per_phase)
    max_change = length_along_path(pp, max_index), max_value

    return mean_per_phase, max_per_phase, max_per_phase_loc, max_change


def length_along_path(pp, index):
    """ Get the lenth measured along the path up to the given index.
    """
    index = min(index, len(pp) - 1)  # beware of the end
    diff_squared = (pp[:index] - pp[1:index+1]) ** 2
    distances = diff_squared.sum(1)
    return (distances ** 0.5).sum()
    # == sum(pp[j].distance(pp[j+1]) for j in range(index))


def get_curvatures(pp):
    """ Get the curvatures on a given path. With pp an Nx3 array
    (i.e. a pointset) returns an array with N values representing
    the curvature in each point.
    """

    # Definition of curvature of a 3D curve:
    #
    # https://en.wikipedia.org/wiki/Curvature#Local_expressions_2
    #
    # ( ( z''y' - y''z' )**2  +  ( x''z' - z''x' )**2  +  ( y''x' - x''y' )**2 )**0.5
    # -------------------------------------------------------------------------------
    #   ( x'**2 + y'**2 + z'**2) ** (3/2)

    # Calculate first derivative in all dimensions
    dx1 = np.gradient(pp[:,0])
    dy1 = np.gradient(pp[:,1])
    dz1 = np.gradient(pp[:,2])

    # And the second order
    dx2 = np.gradient(dx1)
    dy2 = np.gradient(dy1)
    dz2 = np.gradient(dz1)

    # Plug it into the formula
    nom = (
        ( dz2 * dy1 - dy2 * dz1 ) ** 2  +
        ( dx2 * dz1 - dz2 * dx1 ) ** 2  +
        ( dy2 * dx1 - dx2 * dy1 ) ** 2
    ) ** 0.5
    denom = (dx1 ** 2 + dy1 ** 2 + dz1 ** 2) ** (3/2)

    # In the division, avoid nans (and warnings)
    curvatures = np.zeros_like(nom)
    where = nom > 0
    curvatures[where] = nom[where] / denom[where]
    return curvatures
