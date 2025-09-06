# GazeFormer: Text-Guided Multi-Source Fusion for 3D Gaze Estimation

> An experimental research codebase combining CLIP-driven semantic priors, CNN visual descriptors, and a lightweight MoE-enhanced transformer for robust 3D gaze estimation across heterogeneous datasets.

## 🔍 Abstract
We present **GazeFormer**, a modular framework that leverages *frozen vision-language priors* (CLIP), *dataset-agnostic appearance cues*, and *multi-level tokenization* to regress 3D gaze vectors. The system fuses (1) semantic-aligned CLIP embeddings conditioned on illumination, head pose, background, and discrete gaze prototypes; (2) convolutional spatial descriptors; and (3) optional patch & CNN feature tokens via a gated Mixture-of-Experts (MoE) transformer. This design encourages disentanglement between invariant and task-relevant components while retaining generalization across domains (Gaze360, ETH-XGaze, EyeDiap, MPIIFaceGaze). We also include an experimental DeepSeek-style MoE token processor with adaptive top‑k expert routing.

## ✨ Key Contributions
- **Text-conditioned adaptive feature selection:** Dynamic retrieval of attribute tokens (illumination / head pose / background / gaze direction) via CLIP similarity maximization.
- **Dual-path fusion:** Separation of *irrelevant-context-compensated* (feature_1) and *task-aligned* (feature_2) embeddings before geometric regression.
- **Token-level MoE Transformer (experimental):** Modular BlockMoba + MoE layers for multi-source token aggregation (`transformer_models.py` + `model_zhao_test` reference).
- **Flexible dataset adapters:** Unified GazeHub-style parsers with coordinate conversion utilities (CCS ↔ pitch/yaw).
- **Ablation-ready switches:** `ABLA_CONFIG` granular toggles for feature pathways (1–4).

## 🗂 Repository Structure
```
clipmodel.py              # CLIP (modified) components: ViT / ResNet + projection logic
config.py                 # Global hyperparameters, dataset & path configuration
models.py                 # GEWithCLIPModel backbone fusion variants
train_test.py             # Baseline training/eval loop (simplified / synthetic data option)
train.py                  # Advanced training script with MoE transformer integration
transformer_models.py     # MoE + attention experimental blocks (research phase)
gazehub_datasets.py       # Dataset loaders (MPIIFaceGaze / EyeDiap / Gaze360 / ETH-XGaze)
utils.py                  # Geometry + split helpers
loggers.py                # WandB / TensorBoard logging helpers
```

## 🧠 Method Overview
### 1. Semantic Prototype Mining (CLIP)
We pre-tokenize hand-crafted textual descriptors:
- Illumination: {bright, low light, shadow}
- Head pose: {frontal, profile}
- Background: {bright, dark}
- Discrete gaze prototypes: 8 compass-like directional templates

Given an input face crop, CLIP image embeddings are matched to each attribute bank; the highest-similarity text embedding is *added* (residual semantic steering). Two aggregated latent forms emerge:
- `feature_1`: image + (illum + head + bg)
- `feature_2`: image + discrete gaze token

Both are L2-normalized to stabilize scale before fusion.

### 2. Visual Geometry Stream
A CNN backbone (default **ResNet-50**) processes an auxiliary face representation (`other_face`) to produce `feature_3` (flattened spatial encoding). A variant (`GEWithCLIPModel_zhao`) extracts *layer4* activation maps for token-level modeling downstream.

### 3. MoE-Enhanced Token Aggregation (Experimental)
In `train.py`, `process_batch()` constructs a dictionary of tokens:
- `feature_1`, `feature_2`
- `feature_3` (spatial tokens from CNN map)
- `token_img_patch` (ViT patch tokens via a forward hook)
These are projected to a unified `d_model` dimension, concatenated with a learned CLS token, and processed by stacked `BlockMoba` layers (standard attention placeholder + MoE / FF hybrid). Output CLS embedding → 3D gaze regression.

### 4. Losses
- **Angular loss** (primary): acos of normalized vector dot-product
- **Feature separation (optional)**: discourages collapse between `feature_1` and `feature_2`
- (In baseline) CE loss over prototype logits (disabled by default in current script snapshot)

## 🧪 Datasets & Conventions
| Dataset       | Label Mode                  | Coordinate Adaptation                     |
|---------------|-----------------------------|--------------------------------------------|
| Gaze360       | 3D (vector components)      | Direct                                     |
| ETH-XGaze     | Pitch/Yaw (converted)       | `gaze_pitch_yaw_to_ccs` → CCS vector       |
| EyeDiap       | 3D vector (index shift)     | Axis flips conditional on train→test combo |
| MPIIFaceGaze  | 3D vector                   | Same as above                              |

All loaders return: `{ face, other_face }`, `label(3,)` (CCS 3D unit-ish vector; normalization applied inside training loop).

## ⚙️ Configuration
Core settings live in `config.py`:
```
TRAIN_DATASET_NAME = "Gaze360"
TEST_DATASET_NAME  = "Gaze360"  # Cross-dataset eval just switch names
CNN_MODEL = "ResNet-50"         # Options: ResNet-18 / EdgeNeXt-Small
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5  # Overridden in some scripts
DEVICE = cuda | cpu
ABLA_CONFIG = {
  'use_feature_1': True,
  'use_feature_2': True,
  'use_feature_3': True,
  'use_feature_4': True,
}
```

## 🛠 Environment & Dependencies
Minimal (Python ≥3.10):
```
torch
torchvision
timm
easydict
ftfy
regex
opencv-python
numpy
tqdm
wandb
openai-clip   # (pip install git+https://github.com/openai/CLIP.git)
```
Install (example):
```
pip install -r requirements.txt
```
(You may create `requirements.txt` with above list.)

## 🚀 Quick Start
### 1. Prepare Datasets
Organize under:
```
datasets/
  Gaze360/GazeHub/{Image,Label}/
  ETH-XGaze/GazeHub/{Image,Label}/
  ...
```
Expected label format follows GazeHub conventions; see `gazehub_datasets.py` for parsing fields.

### 2. Baseline Training (Synthetic Stub Enabled)
`train_test.py` currently uses a synthetic random loader (for debugging). To restore real training, uncomment the dataset section and remove `random_loader` usage.
```
python train_test.py
```

### 3. Advanced MoE Training
```
python train.py
```
Adjust dataset switch logic inside the script (supports internal train/val split or leave-one-subject-out depending on dataset).

### 4. TensorBoard & Logging
Logs under `out/runs/` (baseline) or `log/` (MoE script). Launch:
```
tensorboard --logdir out/runs
```

## 🧩 Ablation Usage
Toggle feature pathways via `ABLA_CONFIG` (e.g., disable patch tokens or semantic aggregation). This propagates zero tensors to maintain architectural shape without re-wiring.

## 📊 (Placeholder) Expected Results
| Model Variant            | Dataset    | Angular Error (°) | Notes                     |
|-------------------------|-----------|------------------|---------------------------|
| Baseline Fusion         | Gaze360   | ~XX.X            | Placeholder               |
| + Patch Tokens          | Gaze360   | ~XX.X            | Gains if pose diversity   |
| + MoE (DeepSeek-style)  | Gaze360   | ~XX.X            | Slight improvement (est.) |
| Cross → ETH-XGaze       | Gaze360→ETH | ~XX.X           | Domain gap visible        |

(Replace once experiments stabilize.)

## ✅ Code Quality Review (Summary)
| Area | Observation | Recommendation |
|------|-------------|----------------|
| `MixtureOfExperts` | Uses undefined vars in debug print (`j`, `sample_indices`) and concatenation logic misaligns expert outputs | Refactor loop: dispatch per-expert indices, accumulate via `index_add_`; remove/guard prints |
| `VisionTransformer.forward` | Verbose `print()` each batch | Replace with conditional logging flag or `if debug:` |
| `conv1.weight.register_hook` | Always prints grad norm | Gate behind env var; remove for production |
| `features_loss` (baseline) | Divides cosine sim by squared norm product (non-standard) | If intended decorrelation, rename & document; else use standard cosine similarity loss |
| Random loader in `train_test.py` | Obscures real training path | Restore dataset usage before benchmarking |
| Hard-coded text prompts | Could bias domain | Allow YAML/JSON prompt config for reproducibility |
| Hook extraction in `process_batch` | Register/remove each batch | Acceptable; could cache patch tokens if frozen |
| MoE gating prints (commented / partial) | Potential console flood | Remove or use logging module at DEBUG level |
| Global mutable config imports (`from config import *`) | Namespace pollution | Prefer explicit imports or pass config objects |
| Determinism settings mixed | Some flags commented | Provide `--deterministic` CLI flag to toggle |
| `ABLA_CONFIG` | Inconsistent naming vs feature indices | Add docstring mapping feature_x → semantic meaning |
| Error handling for missing files | Minimal | Add asserts for image existence to catch dataset corruption |

## 🐛 Notable Bugs / Risks
1. **`MixtureOfExperts.forward` debug line** references `j` before definition; will raise `NameError` if executed.
2. Potential *mis-weighting* of expert outputs: concatenated expert outputs lose alignment to original ordering → gating weights may not map correctly.
3. No gradient clipping in `train.py` (present in baseline). May cause occasional instability with MoE.
4. Lack of `torch.no_grad()` around frozen backbone feature extraction in places (minor inefficiency).
5. No NaN guard on CLIP similarity logs except partial normalization eps additions.

## 🔧 Suggested Refactors (Future Work)
- Modularize prompt banks into `prompts.yaml`.
- Replace manual similarity argmax with *soft attention weighting* over attributes.
- Introduce learnable *gaze prior codebook* (Vector Quantization) instead of fixed 8 textual tokens.
- Incorporate uncertainty head (predict cone spread / variance).
- Distill MoE transformer into lightweight linear adapter for deployment.

## 📐 Reproducibility Checklist
- [x] Fixed seeds in main scripts (`SEED`)
- [ ] Full cuDNN determinism mode option
- [x] Separate logging for train / eval
- [ ] Requirements file committed
- [ ] Reported commit hash in logs

## ⚠️ Limitations
- Text prompts handcrafted (not automatically adapted cross-domain).
- MoE design exploratory; routing not load-balanced yet.
- Current code mixes baseline & experimental paradigms → needs unification.
- No explicit temporal modeling (for video-based gaze shifts).

## 📄 Citation (Template)
If you build upon this work, please cite:
```
@misc{gazeformer2025,
  title  = {GazeFormer: Text-Guided Multi-Source Fusion for 3D Gaze Estimation},
  author = {Anonymous},
  year   = {2025},
  note   = {Under review}
}
```

## 🤝 Acknowledgements
Uses OpenAI CLIP, TorchVision backbones, and experimental inspiration from mixture-of-experts transformer literature.

---
**Status:** Research prototype (not production-hardened). Please file issues for discrepancies or instability.
