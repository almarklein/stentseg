""" stentseg.motion

For motion calculations on the stent graft.

The motion fields are found using PIRT. This module contains code to
apply the motion to the stent model to make the model dynamic, and also
to visualize the motion in the volume or stent mesh.

"""

from .dynamic import incorporate_motion
from .dynamic import calculate_angle_changes
from .dynamic import get_deform_in_nodes_at_sub_index

from .vis import creat_mesh_with_values
from .vis import convert_mesh_values_to_angle_change
from .vis import remove_stent_from_volume
