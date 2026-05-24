# Anima LoRA Trainer UI

Local and Colab-friendly Gradio UI for training LoRA adapters on
[Anima](https://huggingface.co/circlestone-labs/Anima). It supports both
`kohya-ss/sd-scripts` and `DiffSynth-Studio`, with environment preflight,
run manifests, training history, model/output management, TensorBoard, and
optional sample generation.

## Highlights

- `kohya` and `diffsynth` backends from the same UI.
- DiffSynth auto-clone/install/update support. Users do not need to install DiffSynth manually.
- Training preflight checks for dataset, models, output directory, accelerate, DiffSynth import, metadata, resume path, and `torchao`.
- Stop Training button that terminates the training process tree and marks the run manifest.
- Run manifest per training attempt, saved in the output directory.
- Runs tab for training history.
- Models tab for model status and missing model downloads.
- Outputs tab for LoRA/checkpoint browsing and quick handoff to Sample.
- Samples tab for manual sampling and optional auto-sampling every N epochs.
- Log UI throttling: full logs still go to file, while the browser only keeps a visible tail.

## Requirements

- Python 3.10 or newer.
- NVIDIA GPU with CUDA for training.
- `git`.
- Network access for first-time model and backend downloads.
- Enough disk space for Anima models, training outputs, and optional DiffSynth models.

For Colab, use a GPU runtime and launch with `run_linux.sh`; Colab is detected
automatically and Gradio starts with `share=True`.

## Quick Start

### Linux / Colab / macOS

```bash
bash setup_for_linux.sh
bash run_linux.sh
```

For RTX 5070/5080/5090:

```bash
bash setup_for_linux_rtx5000.sh
bash run_linux.sh
```

### Windows

```bat
setup_for_windows.bat
run_windows.bat
```

For RTX 5070/5080/5090:

```bat
setup_for_windows_rtx5000.bat
run_windows.bat
```

Open the shown Gradio URL, usually `http://127.0.0.1:7860`.

## Dataset Format

Use one flat image folder. Each image should have a matching `.txt` caption:

```text
my_dataset/
  image001.png
  image001.txt
  image002.jpg
  image002.txt
```

Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`.

## Training Workflow

1. Choose `kohya` or `diffsynth` in the top bar.
2. Fill Project Name, Image Directory, Output Directory, Base Model, and GPU.
3. Configure training settings.
4. Click Configure Training.
5. Click Start Training.
6. Use Stop Training to cancel safely if needed.
7. Inspect outputs in the Outputs tab and runs in the Runs tab.

The Training Log box only keeps the newest visible lines to avoid browser lag.
The full log is still written under `logs/` and linked in the run manifest.

## DiffSynth Backend

DiffSynth is treated as a first-class backend:

- The app checks whether `DiffSynth-Studio` exists and whether the current Python can import `diffsynth`.
- If missing, it clones and installs it automatically.
- If an installed `diffsynth` points at another checkout, it reinstalls the local one.
- Old Anima-incompatible LoRA target modules such as `q,k,v,o,ffn.0,ffn.2` are migrated to blank, allowing DiffSynth to auto-detect Anima modules.
- Metadata is generated as `image,prompt`, and DiffSynth args include `--data_file_keys image`.
- `torchao<=0.16.0` is upgraded or removed to avoid PEFT dispatch errors.

For Anima LoRA, leave LoRA Target Modules blank unless you know the exact DiffSynth module names you want.

## Run Manifest

Every Start Training attempt writes a manifest:

```text
<output_dir>/<project>_<backend>_<timestamp>_run_manifest.json
```

It records command, dataset, models, configs, TensorBoard logdir, preflight
results, Python/package versions, status, log path, and output files. Runs that
fail preflight or are cancelled are also tracked.

## Resume / Continue

Advanced Settings includes Resume LoRA / Checkpoint Path.

- `kohya`: written as `network_weights`.
- `DiffSynth`: passed as `--lora_checkpoint`.

The Outputs tab can find the newest LoRA path so you can reuse it in Samples or paste it into resume.

## Samples

The Samples tab can generate preview images with DiffSynth:

- Manual sample: choose a LoRA path or use the latest output.
- Auto sample: enable auto-sampling and set every N epochs.
- Prompt, negative prompt, size, steps, CFG, seed, and low-VRAM mode are configurable.

Auto-sampling runs asynchronously after saved LoRA files are detected. It is off
by default because sampling during training can cause OOM on small GPUs. Generated
images are saved under:

```text
logs/samples/
```

## Model Management

The Models tab shows Anima DiT, Qwen3 text encoder, and VAE status, size, path,
and source URL. Use Download Missing Models to fetch missing files from the UI.

## Training History And Outputs

- Runs tab scans run manifests and shows project, backend, status, latest output, log file, and manifest path.
- Outputs tab scans `.safetensors`, `.pt`, `.pth`, and `.ckpt` files in the output directory.

## TensorBoard

Enable TensorBoard before Configure Training.

- `kohya` uses native TensorBoard logging.
- `DiffSynth` uses a wrapper that patches DiffSynth logging and writes scalar loss events.
- On Colab or remote machines, enable ngrok and provide an auth token if the local iframe is not reachable.

If TensorBoard says no dashboards are active, check that the selected logdir
contains scalar events, not only an empty event file.

## Colab Notes

Recommended Colab flow:

```bash
!git clone <your-repo-url>
%cd citron-anima-lora-trainer-ui
!bash setup_for_linux.sh
!bash run_linux.sh
```

Then open the Gradio share URL. For TensorBoard in Colab, use the TensorBoard
tab with ngrok, or open the printed local/remote URL if your environment exposes
ports.

DiffSynth can download tokenizer/model support files through ModelScope on first
use. The first run may spend time preparing those directories.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Target modules {'v','o','k','q','ffn.0','ffn.2'} not found` | Leave DiffSynth LoRA Target Modules blank and regenerate config. |
| `KeyError: 'prompt'` | Regenerate DiffSynth metadata with the current app; it writes `image,prompt`. |
| `No module named accelerate.__main__` | Use the app's resolved `accelerate launch` path; update app and reconfigure. |
| `torchao 0.10.0 incompatible` | The app auto-upgrades/removes old torchao during DiffSynth setup. Restart the runtime if Python keeps the old package loaded. |
| TensorBoard has no dashboards | Use the exact logdir from Configure Training; DiffSynth scalars appear after loss values are written. |
| UI gets slow from logs | Increase/decrease Visible Log Tail Lines. Full logs remain in `logs/`. |
| Stop button did not instantly release VRAM | Wait a few seconds; on Windows the app calls `taskkill /T`, on Linux it sends SIGTERM to the process group. |
| Auto sample OOM | Disable auto sample or enable low-VRAM sample mode. |

## Project Structure

```text
app.py                         Gradio UI and training orchestration
i18n.py                        English/Chinese UI strings
trainer_core/                  Backend, dataset, progress, manifest, preflight helpers
tools/diffsynth_tensorboard_wrapper.py
tools/anima_sample.py
configs/                       Generated configs and DiffSynth args
logs/                          Training logs, TensorBoard logs, samples
models/anima/                  Anima DiT, Qwen3, VAE files
sd-scripts/                    kohya backend checkout
DiffSynth-Studio/              DiffSynth backend checkout
tests/                         Minimal unit tests
```

## Development Checks

```bash
python -m py_compile app.py i18n.py tools/diffsynth_tensorboard_wrapper.py tools/anima_sample.py trainer_core/*.py tests/test_trainer_core.py
python -m unittest discover -s tests
```

## Credits

- Original Colab notebook: [citronlegacy/citron-colab-anima-lora-trainer](https://github.com/citronlegacy/citron-colab-anima-lora-trainer)
- kohya backend: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- DiffSynth backend: [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- Model: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
