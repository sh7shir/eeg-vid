# EEG Challenge — Baseline v2 (simple, frame-based)

This is a **small update** to the improved organizer baseline, not a new model architecture.

```text
8 video frames
       ↓
frozen AlexNet (ImageNet-1K V1)
       ↓
PCA (fit on training videos only)
       ↓
Ridge regression (fit on training EEG only)
       ↓
predicted EEG for each candidate video
       ↓
correlation with the given test EEG → 102 ranked video IDs
```

## Why this version?

Your earlier four-subject validation results were:

| Subject | Top-1 | Top-5 |
|---|---:|---:|---:|
| 01 | 3.92% | 10.8% |
| 02 | 4.9% | 22.5% |
| 03 | 5.88% | 13.73% |
| 04 | 2.94% | 8,82% |
| Mean | 5.39% | 20.10% |

These results exceed random chance (0.98% top-1 and 4.90% top-5).
The organizers' introductory example reported roughly 4% top-1 and 11.7% top-5, but a fair head-to-head comparison requires running both implementations on identical test rows and subjects. Small top-1 differences are noisy with just 102 trials per subject.

Two *optional*, validated changes are added:

1. **`halves` video representation:** average frames 1–4 and 5–8 separately. Unlike averaging all frames, it retains a little temporal order. These are **CNN frame features**, never raw pixel vectors.
2. **`center_trials=True` similarity:** subtract the common response from all test EEG trials (for each channel/time sample), and likewise subtract the candidate-averaged prediction. This is unsupervised test-data centering, expressly permitted by the challenge. It is **not** training on test recordings and does not assume unique video assignments.

We keep the old `mean` and `frames` representations and the old uncentered scoring as valid candidates. Notebook 01 compares all of them on subjects 01–04, then selects the best common configuration using 70% top-1 / 30% top-5. If the old approach wins again, it stays selected automatically.

**Actual improved accuracy is unknown until you run the notebooks on your recordings.** No actual challenge EEG/video arrays were attached to the conversation.

## Run it in your existing Windows project

Keep the original folder structure:

```text
D:\eegx\eeg\
    01_preprocess_and_train_model.ipynb
    02_predict_movies.ipynb
    frame_baseline.py
    requirements.txt
    stimuli_train\       0001.mp4 ... 1000.mp4
    stimuli_test\        1001.mp4 ... 1102.mp4
    trn\                 sub-01_trn.npz ... sub-06_trn.npz
    tst\                 sub-01_tst.npy ... sub-06_tst.npy
    tst_stim\            sub-01_tst_stim.npy ... sub-04_tst_stim.npy
```

In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m notebook
```

Run notebook **01**, then notebook **02**. The first pass downloads the public pretrained AlexNet ImageNet-1K V1 weights. Video features are stored in `cache/*.npz`; training parameters are stored under `models/`.

Notebook 02 creates exactly:

- `sub-05_ranked_candidate_list.npy`
- `sub-06_ranked_candidate_list.npy`

Each is a NumPy integer array with shape `(102, 102)`. Every row is a full permutation of the IDs `1001` through `1102`. The row order matches the unsorted source file `tst/sub-XX_tst.npy`. No Hungarian assignment, test-label templates, training on test EEG, or supervised use of development test recordings.

## Reproducibility

- Python 3.11.
- Pretrained model: `torchvision.models.AlexNet_Weights.IMAGENET1K_V1` (public, pretrained on ImageNet).
- Save the exact torch and torchvision package versions from `models/selection.json` with your report. For a full environment lock, run `python -m pip freeze > installed_versions.txt`.
- Commit notebook files, helper, requirements, `models/*.joblib` and `models/selection.json` to your reproducible repo. Do not commit videos/frames.
- To test logic without any challenge media, run `python test_small_pipeline.py`.

## Fair improvement check

Notebook 01 prints the mean validation top-1 and top-5 for **all 12 configurations** (3 frame modes × 2 EEG windows × 2 scoring choices). Compare the best row with the earlier **5.39% top-1 / 20.10% top-5**. A small gain over these measurements should be treated cautiously because the same validation labels also select the configuration; keep detailed per-subject results and ideally compare against a fully frozen reference on a second train-only validation split.
