from pathlib import Path

import pytest

from wuji_glove_d435_ftp1_collect import FTP1Collector


def test_episode_layout_rejects_numeric_participant_before_creating_directories(tmp_path: Path):
    collector = FTP1Collector.__new__(FTP1Collector)
    collector.output_dir = tmp_path
    with pytest.raises(ValueError, match="非纯数字"):
        collector.select_episode_layout("1", "t1", "有手套")
    assert list(tmp_path.iterdir()) == []


def test_episode_layout_accepts_open_ended_task_number(tmp_path: Path):
    collector = FTP1Collector.__new__(FTP1Collector)
    collector.output_dir = tmp_path
    collector._install_reader = lambda: None

    collector.select_episode_layout("zy", "t15", "有手套")

    assert collector.task_id == "t15"
    assert collector.output_dir == tmp_path / "zy" / "t15" / "有手套"
    assert (tmp_path / "zy" / "t15" / "无手套").is_dir()
    assert not (tmp_path / "zy" / "t14").exists()


@pytest.mark.parametrize("task_id", ["t0", "t01", "t-1", "t", "15"])
def test_episode_layout_rejects_noncanonical_task_number(
    tmp_path: Path, task_id: str
):
    collector = FTP1Collector.__new__(FTP1Collector)
    collector.output_dir = tmp_path
    with pytest.raises(ValueError, match="正整数"):
        collector.select_episode_layout("zy", task_id, "有手套")
