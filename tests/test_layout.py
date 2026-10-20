from pathlib import Path
import json
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTest(unittest.TestCase):
    def test_four_workflows_exist(self):
        workflows = ROOT / "workflows"
        self.assertEqual(
            {path.name for path in workflows.iterdir() if path.is_dir()},
            {"pretrain", "representation_eval", "planar", "robot"},
        )

    def test_workflow_documentation_is_present(self):
        documents = (
            "workflows/pretrain/README.md",
            "workflows/pretrain/scripts/build_kinetics_lmdb.py",
            "workflows/representation_eval/README.md",
            "workflows/planar/README.md",
            "workflows/robot/README.md",
        )
        for relative in documents:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_release_environment_pins_current_pytorch(self):
        requirements = (ROOT / "requirement.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.2.2+cu118", requirements)
        self.assertIn("torchvision==0.17.2+cu118", requirements)
        self.assertIn("xformers==0.0.25.post1+cu118", requirements)

    def test_shared_core_is_organized_by_capability(self):
        package = ROOT / "src" / "slot_jepa"
        self.assertFalse((package / "vjsepa").exists())
        self.assertFalse((package / "_bootstrap.py").exists())
        self.assertFalse((package / "transport").exists())
        for relative in (
            "encoders/factory.py",
            "slots/extractor.py",
            "predictors/action_mask.py",
            "losses/reconstruction.py",
            "losses/transport.py",
            "data/kinetics_video.py",
            "utils/logging.py",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((package / relative).is_file())

    def test_removed_artifact_links_are_absent(self):
        removed_links = (
            "workflows/planar/checkpoints",
            "workflows/planar/sampled_data",
            "workflows/planar/slot-ac/outputs",
            "workflows/planar/slot-ac/context_cache",
            "workflows/robot/outputs",
            "workflows/robot/realm/successful_case",
        )
        for relative in removed_links:
            with self.subTest(relative=relative):
                path = ROOT / relative
                self.assertFalse(path.is_symlink())
                self.assertFalse(path.exists())

    def test_internal_experiment_code_is_slot_jepa(self):
        old_code = "v2.1-stu_" + "vjepa-tea_dinov2"
        old_identifier = "v2.1_stu_" + "vjepa_tea_dinov2"
        checkpoint_root = ROOT / "artifacts" / "checkpoints"
        self.assertTrue((checkpoint_root / "pretrain" / "slot_jepa_pretrained.pt").is_file())
        self.assertTrue((checkpoint_root / "pretrain" / "slot_jepa_pretrained.yaml").is_file())
        self.assertTrue((checkpoint_root / "downstream" / "robot" / "slot_jepa_droid_finetuned.pt").is_file())
        self.assertTrue((checkpoint_root / "downstream" / "robot" / "slot_jepa_droid_finetuned.yaml").is_file())
        for task in ("TwoRoom", "Maze", "PushT", "TwoRoomR", "MazeR", "PushObj"):
            with self.subTest(planar_config=task):
                task_root = checkpoint_root / "downstream" / "planar" / "outputs" / task
                self.assertTrue((task_root / "hydra.yaml").is_file())
                self.assertTrue((task_root / "checkpoints" / "model_latest.pth").is_file())
        self.assertFalse((ROOT / "workflows" / "representation_eval" / "checkpoints").exists())

        text_suffixes = {".log", ".md", ".py", ".yaml", ".yml"}
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in text_suffixes:
                with self.subTest(path=path.relative_to(ROOT)):
                    content = path.read_text(encoding="utf-8")
                    self.assertNotIn(old_code, content)
                    self.assertNotIn(old_identifier, content)

    def test_exactly_six_planar_checkpoints_are_released(self):
        data_root = ROOT / "artifacts" / "checkpoints" / "downstream" / "planar" / "outputs"
        self.assertEqual(
            {path.name for path in data_root.iterdir() if path.is_dir()},
            {
                "TwoRoom",
                "Maze",
                "PushT",
                "TwoRoomR",
                "MazeR",
                "PushObj",
            },
        )
        for task in ("TwoRoom", "Maze", "PushT", "TwoRoomR", "MazeR", "PushObj"):
            with self.subTest(task=task):
                self.assertTrue((data_root / task / "hydra.yaml").is_file())
                self.assertTrue((data_root / task / "checkpoints" / "model_latest.pth").is_file())

    def test_minimal_robot_goal_package_is_released(self):
        goal_root = ROOT / "artifacts" / "data" / "robot" / "successful_case"
        cases = sorted(path for path in goal_root.iterdir() if path.is_dir())
        self.assertEqual(len(cases), 108)
        for case in cases:
            with self.subTest(case=case.name):
                manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
                expected = {"manifest.json", "goal_targets.pt"}
                for target_name in manifest["target_names"]:
                    expected.update(
                        {
                            f"{target_name}_base.png",
                            f"{target_name}_second.png",
                            f"{target_name}_wrist.png",
                        }
                    )
                self.assertEqual(
                    {path.name for path in case.iterdir() if path.is_file()},
                    expected,
                )

    def test_planar_entrypoints_use_paper_task_names(self):
        config_root = ROOT / "workflows" / "planar" / "conf"
        tasks = {"TwoRoom", "Maze", "PushT", "TwoRoomR", "MazeR", "PushObj"}
        self.assertEqual(
            {path.stem.removeprefix("train_") for path in config_root.glob("train_*.yaml")},
            tasks,
        )
        self.assertEqual(
            {
                path.stem.removeprefix("plan_")
                for path in config_root.glob("plan_*.yaml")
                if path.stem != "plan_base"
            },
            tasks,
        )
        for task in tasks:
            with self.subTest(task=task):
                train_text = (config_root / f"train_{task}.yaml").read_text(encoding="utf-8")
                plan_text = (config_root / f"plan_{task}.yaml").read_text(encoding="utf-8")
                self.assertIn(f"outputs/{task}", train_text)
                self.assertIn(f"model_name: {task}", plan_text)
                self.assertIn(f"plan_outputs/{task}", plan_text)

    def test_artifacts_contain_only_release_packages(self):
        artifact_root = ROOT / "artifacts"
        self.assertEqual(
            {path.name for path in artifact_root.iterdir()},
            {"checkpoints", "color_channel_weights", "data"},
        )
        color_weights = artifact_root / "color_channel_weights"
        self.assertTrue((color_weights / "README.md").is_file())
        self.assertTrue(
            (color_weights / "planar_block12_important_objects_v2.pt").is_file()
        )
        for excluded in ("training", "caches", "evaluations"):
            self.assertFalse((artifact_root / excluded).exists())

        planar_data = artifact_root / "data" / "planar"
        self.assertEqual(
            {path.name for path in planar_data.iterdir() if path.is_dir()},
            {"TwoRoom", "Maze", "PushT", "TwoRoomR", "MazeR", "PushObj"},
        )
        for task in ("TwoRoom", "Maze", "PushT", "TwoRoomR", "MazeR", "PushObj"):
            with self.subTest(planar_data=task):
                self.assertTrue((planar_data / task / "targets" / "test.pkl").is_file())

    def test_evaluation_outputs_are_not_bundled(self):
        excluded = (
            "artifacts/evaluations",
            "workflows/representation_eval/outputs",
            "workflows/planar/plan_outputs",
            "workflows/robot/realm/outputs",
        )
        for relative in excluded:
            with self.subTest(relative=relative):
                path = ROOT / relative
                self.assertFalse(path.exists())
                self.assertFalse(path.is_symlink())

    def test_visual_evaluation_keeps_only_the_released_slot_mix(self):
        config_root = ROOT / "workflows" / "representation_eval" / "configs" / "tasks"
        released = {
            "depth/kitti_slot_jepa.yaml",
            "depth/nyuv2_slot_jepa.yaml",
            "depth/sunrgbd_transfer_nyuv2_slot_jepa.yaml",
            "linear_seg/ade20k_voc_slot_jepa.yaml",
            "linear_seg/coco_slot_jepa.yaml",
            "vos/davis_slot_jepa.yaml",
        }
        self.assertEqual(
            {
                str(path.relative_to(config_root))
                for path in config_root.rglob("*_slot_jepa.yaml")
            },
            released,
        )

        for config_path in config_root.rglob("*.yaml"):
            with self.subTest(config=config_path.relative_to(config_root)):
                self.assertNotRegex(config_path.name, r"(?:alpha|slot)[_-]?0\.[0-9]")
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                model = config.get("model", {})
                if "slot_alpha" in model:
                    self.assertEqual(model["slot_alpha"], 0.3)

        for relative in released:
            config = yaml.safe_load((config_root / relative).read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["slot_alpha"], 0.3)
            self.assertNotIn("0.3", config["experiment"]["name"])
            self.assertNotIn("0.3", config["experiment"]["output_subdir"])


if __name__ == "__main__":
    unittest.main()
