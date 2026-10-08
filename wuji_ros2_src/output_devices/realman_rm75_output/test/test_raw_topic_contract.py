from pathlib import Path


PICO_NODE = (
    Path(__file__).resolve().parents[3]
    / "input_devices/pico_input/pico_input/pico_input_node.py"
)


def test_raw_topic_is_published_before_incremental_controller():
    source = PICO_NODE.read_text(encoding="utf-8")
    process_start = source.index("    def _process_trackers(")
    process_end = source.index("    def _publish_raw_tracker_pose(", process_start)
    process_body = source[process_start:process_end]
    raw_call = process_body.index("self._publish_raw_tracker_pose(")
    incremental_call = process_body.index("self.controller.compute_target_pose(pose, role)")
    assert raw_call < incremental_call
    assert process_body.index("if not tracker_data.is_valid:") < raw_call
    assert process_body.index("role = self._get_role_for_tracker(tracker_data)") < raw_call


def test_raw_topic_message_contract():
    source = PICO_NODE.read_text(encoding="utf-8")
    assert "'/pico/right_wrist/raw_pose'" in source
    assert "'/pico/right_arm/raw_pose'" in source
    method = source[source.index("    def _publish_raw_tracker_pose("):
                    source.index("    def _broadcast_tf(")]
    assert "msg.header.frame_id = 'pico_tracking'" in method
    assert "tracker_data.position[0]" in method
    assert "tracker_data.orientation[0]" in method
    assert "tracker_data.orientation[3]" in method
    forbidden_transforms = [
        "compute_target_pose", "pico_to_robot", "robot_init",
        "position_scale", "OneEuro",
    ]
    assert not any(name in method for name in forbidden_transforms)


def test_existing_tianji_target_path_still_uses_incremental_result():
    source = PICO_NODE.read_text(encoding="utf-8")
    assert "pos, quat = self.controller.compute_target_pose(pose, role)" in source
    assert "self._publish_pose(self.right_arm_pose_pub, now, parent_frame, pos, quat)" in source
    assert "parent_frame = 'world_right'" in source
