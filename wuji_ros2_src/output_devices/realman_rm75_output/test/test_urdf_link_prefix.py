from xml.etree import ElementTree

import pytest

from realman_rm75_output.urdf_link_prefix import prefix_urdf_link_names


SIMPLE_URDF = """
<robot name="sample">
  <link name="base_link"/>
  <link name="Link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="Link1"/>
  </joint>
  <gazebo reference="Link1"/>
</robot>
"""


def test_prefixes_link_frames_but_preserves_joint_names():
    root = ElementTree.fromstring(
        prefix_urdf_link_names(SIMPLE_URDF, "rm75_command_"))

    assert [link.get("name") for link in root.findall("link")] == [
        "rm75_command_base_link",
        "rm75_command_Link1",
    ]
    joint = root.find("joint")
    assert joint.get("name") == "joint1"
    assert joint.find("parent").get("link") == "rm75_command_base_link"
    assert joint.find("child").get("link") == "rm75_command_Link1"
    assert root.find("gazebo").get("reference") == "rm75_command_Link1"


def test_rejects_empty_prefix():
    with pytest.raises(ValueError):
        prefix_urdf_link_names(SIMPLE_URDF, "")


def test_can_recolor_the_second_robot_for_unambiguous_rviz_display():
    source = SIMPLE_URDF.replace(
        '<link name="Link1"/>',
        '<link name="Link1"><visual><geometry/></visual></link>')
    root = ElementTree.fromstring(prefix_urdf_link_names(
        source, "rm75_command_", "0.1 0.85 0.25 1"))

    color = root.findall("link")[1].find("visual/material/color")
    assert color.get("rgba") == "0.1 0.85 0.25 1"
