import csv
import tempfile
import unittest
from pathlib import Path

from trainer_core import diffsynth, diffsynth_support, steps
from trainer_core.backends import build_diffsynth_run_spec, build_kohya_run_spec
from trainer_core.catalog import model_status_rows, scan_output_files, scan_run_manifests
from trainer_core.dataset import generate_diffsynth_metadata, migrate_diffsynth_metadata_for_anima
from trainer_core.manifest import create_run_manifest
from trainer_core.progress import ProgressTracker, parse_structured_progress, parse_tqdm_progress
from trainer_core.sample_queue import format_sample_elapsed, is_terminal_sample_status


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

    def test_sample_queue_status_and_elapsed_formatting(self):
        self.assertTrue(is_terminal_sample_status("done"))
        self.assertTrue(is_terminal_sample_status("failed:1"))
        self.assertFalse(is_terminal_sample_status("running"))
        self.assertEqual(format_sample_elapsed(5), "5s")
        self.assertEqual(format_sample_elapsed(65), "1m 5s")
        self.assertEqual(format_sample_elapsed(3661), "1h 1m 1s")

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
                tokenizer_path=str(root / "Qwen3-0.6B"),
                tokenizer_t5xxl_path=str(root / "tokenizer_3"),
            )

        self.assertIn("--lora_checkpoint", args)
        self.assertEqual(args[args.index("--tokenizer_path") + 1], str(root / "Qwen3-0.6B"))
        self.assertEqual(args[args.index("--tokenizer_t5xxl_path") + 1], str(root / "tokenizer_3"))
        self.assertEqual(args[args.index("--lora_rank") + 1], "20")
        self.assertEqual(args[args.index("--gradient_accumulation_steps") + 1], "1")
        self.assertNotIn("--network_alpha", args)
        self.assertNotIn("--train_batch_size", args)

    def test_diffsynth_args_migration_adds_tokenizer_paths(self):
        args = diffsynth.set_anima_tokenizer_args(
            ["--model_paths", "[]"],
            tokenizer_path="/models/Qwen/Qwen3-0.6B",
            tokenizer_t5xxl_path="/models/sd35/tokenizer_3",
        )

        self.assertEqual(args[args.index("--tokenizer_path") + 1], "/models/Qwen/Qwen3-0.6B")
        self.assertEqual(args[args.index("--tokenizer_t5xxl_path") + 1], "/models/sd35/tokenizer_3")

    def test_progress_tracker_accumulates_reset_epochs(self):
        tracker = ProgressTracker(expected_total=30, expected_epoch_total=20)

        self.assertEqual(parse_tqdm_progress("10/20 [00:01<00:01]"), (10, 20))
        self.assertEqual(
            tracker.feed("10/20 [00:01<00:01]"),
            "[progress] epoch 1/2: 10/20 (50.0%), total 10/30 (33.3%)",
        )
        self.assertEqual(
            tracker.feed("20/20 [00:02<00:00]"),
            "[progress] epoch 1/2: 20/20 (100.0%), total 20/30 (66.7%)",
        )
        self.assertEqual(tracker.feed("1/10 [00:00<00:01]"), None)

    def test_progress_tracker_ignores_non_training_tqdm_totals(self):
        tracker = ProgressTracker(expected_total=8220, expected_epoch_total=822)

        self.assertIsNone(tracker.feed("100/1000 [00:01<00:09]"))
        self.assertEqual(
            tracker.feed("528/822 [04:27<02:30, 1.96it/s]"),
            "[progress] epoch 1/10: 528/822 (64.2%), total 528/8220 (6.4%)",
        )

    def test_progress_tracker_prefers_structured_diffsynth_events(self):
        tracker = ProgressTracker(expected_total=8220, expected_epoch_total=822)
        line = '__ANIMA_PROGRESS__{"source":"diffsynth_logger","step":823,"total":8220,"epoch":2,"epochs":10,"epoch_step":1,"epoch_total":822}'

        event = parse_structured_progress(line)
        self.assertIsNotNone(event)
        self.assertEqual(event["step"], 823)
        self.assertEqual(
            tracker.feed(line),
            "[progress:diffsynth_logger] epoch 2/10: 1/822 (0.1%), total 823/8220 (10.0%)",
        )
        self.assertIsNone(tracker.feed("100/1000 [00:01<00:09]"))

    def test_diffsynth_parameter_rows_mark_kohya_only_settings(self):
        rows = diffsynth.parameter_rows_from_args(["--lora_rank", "20", "--learning_rate", "0.0001"])
        by_name = {row["parameter"]: row for row in rows}

        self.assertEqual(by_name["LoRA rank"]["value"], "20")
        self.assertEqual(by_name["Learning rate"]["value"], "0.0001")
        self.assertEqual(by_name["Train Batch Size"]["status"], "not used by DiffSynth")

    def test_diffsynth_support_paths_and_directory_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = diffsynth_support.SUPPORT_SPECS[0]
            path = diffsynth_support.support_path(spec, root)
            self.assertFalse(diffsynth_support.is_support_ready(spec, root))
            path.mkdir(parents=True)
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

            self.assertTrue(diffsynth_support.is_support_ready(spec, root))
            self.assertGreater(diffsynth_support.path_size_bytes(path), 0)

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
