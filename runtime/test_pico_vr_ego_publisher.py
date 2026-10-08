import pytest

from pico_vr_ego_publisher import _eye_video_filter, _parse_roi, _parse_screen_size


def test_policy_roi_is_eye_local_and_keeps_ui_out_of_stream():
    roi = _parse_roi("520,180,300,600")
    assert roi == (520, 180, 300, 600)
    assert _eye_video_filter("right", roi) == "crop=iw/2:ih:iw/2:0,crop=300:600:520:180"


def test_both_eyes_are_cropped_before_side_by_side_composition():
    roi = _parse_roi("520,180,300,600")
    assert _eye_video_filter("both", roi) == (
        "split=2[left_src][right_src];"
        "[left_src]crop=iw/2:ih:0:0,crop=300:600:520:180[left_eye];"
        "[right_src]crop=iw/2:ih:iw/2:0,crop=300:600:520:180[right_eye];"
        "[left_eye][right_eye]hstack=inputs=2:shortest=1[stereo]"
    )


def test_raw_roi_can_be_selected_for_diagnostics():
    assert _parse_roi("raw") is None
    assert _parse_roi("none") is None


def test_screen_size_requires_even_stereo_width():
    assert _parse_screen_size("1920x960") == (1920, 960)
    with pytest.raises(ValueError):
        _parse_screen_size("1919x960")



@pytest.mark.parametrize("value", ["520,80,400", "x,80,400,800", "-1,0,10,10", "0,0,0,10"])
def test_roi_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _parse_roi(value)
