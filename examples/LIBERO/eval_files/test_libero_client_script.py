import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class LiberoClientScriptTest(unittest.TestCase):
    def test_suite_loop_uses_script_selected_by_run_mode(self):
        script = (REPO_ROOT / "libero_client.zsh").read_text()

        active_lines = [
            line.strip()
            for line in script.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        suite_invocations = [
            line for line in active_lines if line.startswith("TASK_SUITE_NAME=")
        ]

        self.assertTrue(suite_invocations, "No active LIBERO suite invocation found")
        self.assertTrue(
            all('bash "${RUN_SCRIPT}"' in line for line in suite_invocations),
            f"Suite invocation bypasses RUN_MODE selection: {suite_invocations}",
        )

    def test_collection_script_passes_independent_collection_metadata(self):
        script = (REPO_ROOT / "examples/LIBERO/eval_files/collect_libero_dataset.sh").read_text()
        self.assertIn('--args.collection-id "${COLLECTION_ID}"', script)
        self.assertIn('--args.seed-namespace "${SEED_NAMESPACE}"', script)
        self.assertIn('--args.seed "${SEED}"', script)
        self.assertIn('--args.episode-start-index "${EPISODE_START_INDEX}"', script)
        self.assertIn('${CKPT_STEM}_${COLLECTION_ID}_${TASK_SUITE_NAME}.hdf5', script)


if __name__ == "__main__":
    unittest.main()
