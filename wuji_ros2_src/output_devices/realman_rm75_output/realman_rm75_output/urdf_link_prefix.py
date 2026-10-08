"""Create an independent TF tree for a second copy of a URDF robot."""

from xml.etree import ElementTree


def prefix_urdf_link_names(
        urdf_xml: str, prefix: str, visual_rgba: str | None = None) -> str:
    """Prefix link names and every URDF reference to those links.

    Joint names intentionally remain unchanged so a normal JointState message
    containing ``joint1`` ... ``joint7`` can drive either copy of the robot.
    """
    if not prefix:
        raise ValueError("link prefix must not be empty")

    root = ElementTree.fromstring(urdf_xml)
    links = root.findall("link")
    old_names = [link.get("name") for link in links]
    if any(name is None or name == "" for name in old_names):
        raise ValueError("every URDF link must have a non-empty name")

    name_map = {name: f"{prefix}{name}" for name in old_names}
    for link in links:
        link.set("name", name_map[link.get("name")])
        if visual_rgba is not None:
            for visual in link.findall("visual"):
                material = visual.find("material")
                if material is None:
                    material = ElementTree.SubElement(visual, "material")
                color = material.find("color")
                if color is None:
                    color = ElementTree.SubElement(material, "color")
                color.set("rgba", visual_rgba)

    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            element = joint.find(tag)
            if element is None:
                continue
            old_name = element.get("link")
            if old_name in name_map:
                element.set("link", name_map[old_name])

    # These references are not used by robot_state_publisher, but keeping them
    # consistent makes the generated URDF safe for other consumers too.
    for gazebo in root.findall("gazebo"):
        reference = gazebo.get("reference")
        if reference in name_map:
            gazebo.set("reference", name_map[reference])

    root.set("name", f"{prefix}{root.get('name', 'robot')}")
    return ElementTree.tostring(root, encoding="unicode")
