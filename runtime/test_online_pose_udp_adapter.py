import json
import tempfile
import unittest
from pathlib import Path

from online_pose_udp_adapter import session_checkpoint


class SessionModelTests(unittest.TestCase):
    def status(self, root, metadata):
        session = root / "session"
        session.mkdir()
        (session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
        return {"session_id": metadata["session_id"], "output_directory": str(session)}

    def test_five_finger_directory_is_accepted_and_identified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "cal120_seed2026"
            weights.mkdir()
            files = []
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                path = weights / f"finger_{finger}.pt"
                path.touch()
                files.append(str(path))
            status = self.status(root, {
                "session_id": "finger-session",
                "backend": "fingers_wide",
                "model_directory": str(weights),
                "weight_files": files,
                "model_sha256": "ensemble-digest",
            })
            path, digest, backend = session_checkpoint(
                status, require_personal=False, require_finger_model=True)
            self.assertEqual(path, weights.resolve())
            self.assertEqual(digest, "ensemble-digest")
            self.assertEqual(backend, "fingers_wide")
            with self.assertRaisesRegex(RuntimeError, "not a personal"):
                session_checkpoint(status, require_personal=True)

    def test_personal_checkpoint_still_uses_existing_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "personal_v6.pt"
            checkpoint.touch()
            status = self.status(root, {
                "session_id": "personal-session",
                "checkpoint": str(checkpoint),
                "base_sha256": "personal-digest",
            })
            path, digest, backend = session_checkpoint(
                status, require_personal=True)
            self.assertEqual(path, checkpoint.resolve())
            self.assertEqual(digest, "personal-digest")
            self.assertEqual(backend, "online_personal")
            with self.assertRaisesRegex(RuntimeError, "not using the five-finger"):
                session_checkpoint(status, require_personal=False,
                                   require_finger_model=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
