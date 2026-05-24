import csv
import tempfile
import unittest
from pathlib import Path

from trainer_core import diffsynth, steps
from trainer_core.backends import build_diffsynth_run_spec, build_kohya_run_spec
from trainer_core.catalog import model_status_rows, scan_output_files, scan_run_manifests
from trainer_core.dataset import generate_diffsynth_metadata, migrate_diffsynth_metadata_for_anima
from trainer_core.manifest import create_run_manifest
from trainer_core.progress import ProgressTracker, parse_tqdm_progress


class TrainerCoreTests(unittest.TestCase):
    def test_diffsynth_step_estimate_uses_repeated_samples(self):
        estimate = steps.estimate_steps(
            backend="diffsynth",
            n_images=411,
            repeats=10,
            dataset_repeat=5,
            epochs=10,
            train_batch_size=4,
            gradient_accumulation_steps=1,
        )

        self.assertEqual(estimate["progress_per_epoch"], 2055)
        self.assertEqual(estimate["progress_total"], 20550)
        self.assertEqual(estimate["optimizer_total"], 20550)

    def test_legacy_diffsynth_lora_targets_migrate_to_anima_defaults(self):
        self.assertEqual(diffsynth.normalize_lora_target_modules("q,k,v,o,ffn.0,ffn.2"), "")

    def test_diffsynth_args_migration_adds_data_file_keys(self):
        args = diffsynth.migrate_args_for_anima([
            "--dataset_metadata_path",
            "missing.csv",
            "--lora_target_modules",
            "q,k,v,o,ffn.0,ffn.2",
        ])

        self.assertEqual(args[0:2], ["--data_file_keys", "image"])
        self.assertEqual(args[args.index("--lora_target_modules") + 1], "")

    def test_metadata_generation_and_migration_use_anima_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.png").write_bytes(b"fake")
            (root / "sample.txt").write_text("a girl, guitar", encoding="utf-8")

            metadata_path, rows = generate_diffsynth_metadata(root, root / "metadata.csv")
            self.assertEqual(rows, 1)
            with open(metadata_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["image", "prompt"])
                self.assertEqual(next(reader)["prompt"], "a girl, guitar")

            legacy_path = root / "legacy.csv"
            with open(legacy_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
                writer.writeheader()
                writer.writerow({"file_name": "sample.png", "text": "legacy prompt"})

            migrated = migrate_diffsynth_metadata_for_anima(legacy_path)
            with open(migrated, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["image", "prompt"])
                self.assertEqual(next(reader)["prompt"], "legacy prompt")

    def test_diffsynth_training_args_include_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _ = diffsynth.create_training_args(
                args_path=root / "args.json",
                output_dir=str(root / "out"),
                dit_model_path=root / "dit.safetensors",
                qwen3_model_path=root / "qwen.safetensors",
                vae_model_path=root / "vae.safetensors",
                image_dir=str(root / "images"),
                metadata_csv=str(root / "metadata.csv"),
                learning_rate=0.0001,
                max_train_epochs=2,
                dataset_repeat=5,
                max_pixels=1048576,
                lora_rank=20,
                lora_target_modules="",
                use_gradient_checkpointing=True,
                gradient_accumulation_steps=1,
                save_steps=0,
                resume_lora_path=str(root / "resume.safetensors"),
            )

        self.assertIn("--lora_checkpoint", args)
        self.assertEqual(args[args.index("--lora_rank") + 1], "20")
        self.assertEqual(args[args.index("--gradient_accumulation_steps") + 1], "1")
        self.assertNotIn("--network_alpha", args)
        self.assertNotIn("--train_batch_size", args)

    def test_progress_tracker_accumulates_reset_epochs(self):
        tracker = ProgressTracker(expected_total=30)

        self.assertEqual(parse_tqdm_progress("10/20 [00:01<00:01]"), (10, 20))
        self.assertEqual(tracker.feed("10/20 [00:01<00:01]"), "[progress] 10/30 (33.3%)")
        self.assertEqual(tracker.feed("20/20 [00:02<00:00]"), "[progress] 20/30 (66.7%)")
        self.assertEqual(tracker.feed("1/10 [00:00<00:01]"), "[progress] 21/30 (70.0%)")

    def test_backend_specs_build_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kohya = build_kohya_run_spec(
                accelerate_launch=["accelerate", "launch"],
                accelerate_config="acc.yaml",
                threads=1,
                gpu_idx="0",
                train_script=root / "train.py",
                train_config="train.toml",
                dataset_config="dataset.toml",
                cwd=root,
            )
            self.assertIn("--dataset_config", kohya.command)
            self.assertEqual(kohya.backend, "kohya")

            ds = build_diffsynth_run_spec(
                accelerate_launch=["accelerate", "launch"],
                accelerate_config="acc.yaml",
                threads=1,
                gpu_idx="0",
                train_script=root / "ds_train.py",
                train_entrypoint=root / "wrapper.py",
                train_args=["--dataset_metadata_path", "metadata.csv"],
                args_path="args.json",
                metadata_path="metadata.csv",
                cwd=root,
            )
            self.assertIn(str(root / "wrapper.py"), ds.command)
            self.assertEqual(ds.configs["metadata"], "metadata.csv")

    def test_catalog_scans_outputs_models_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            lora = output / "demo.safetensors"
            lora.write_bytes(b"weights")
            manifest = create_run_manifest(
                output_dir=str(output),
                project_name="demo",
                backend="diffsynth",
                command=["python", "train.py"],
                dataset={},
                models={},
                configs={},
                training={},
                tensorboard={},
                preflight=[],
            )

            outputs = scan_output_files(output)
            self.assertEqual(outputs[0]["name"], "demo.safetensors")
            manifests = scan_run_manifests(output)
            self.assertEqual(manifests[0]["manifest"], manifest)
            models = model_status_rows({"LoRA": str(lora)}, {"LoRA": "https://example.test/model"})
            self.assertEqual(models[0]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
