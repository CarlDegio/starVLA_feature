from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "collect_rejection_test.sh"


class CollectRejectionTestScriptTest(unittest.TestCase):
    def test_script_collects_exact_four_suite_hdf5_bundle_without_videos(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        suites_match = re.search(r"SUITES=\(([^)]+)\)", source)
        self.assertIsNotNone(suites_match)
        self.assertEqual(
            suites_match.group(1).split(),
            ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        )
        self.assertIn('EPISODES_PER_SUITE="${EPISODES_PER_SUITE:-100}"', source)
        self.assertIn('EPISODE_START_INDEX="${EPISODE_START_INDEX:-10}"', source)
        self.assertIn("TASKS_PER_SUITE=10", source)
        self.assertIn('NUM_TRIALS_PER_TASK=$((EPISODES_PER_SUITE / TASKS_PER_SUITE))', source)
        self.assertIn('MAX_TASKS="${TASKS_PER_SUITE}"', source)
        self.assertIn('EPISODE_START_INDEX="${EPISODE_START_INDEX}"', source)
        self.assertIn('${COLLECTION_ID}_${suite}.hdf5', source)
        self.assertIn('bash "${COLLECTOR}"', source)
        self.assertNotIn("video_out_path", source)

    def test_script_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
