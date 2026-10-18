"""Small, leak-free extension of the organizers' AlexNet -> PCA -> Ridge EEG baseline.

Pretrained AlexNet processes each *frame*, not raw pixel vectors. Video-to-EEG
regressors are fitted strictly on labeled training recordings.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

N_FRAMES = 8
DECIMATION = 5
EEG_HZ = 100  # source signals are 500 Hz
TEST_IDS = np.arange(1001, 1103, dtype=np.int64)
WEIGHTS_NAME = 'torchvision.models.AlexNet_Weights.IMAGENET1K_V1'
FEATURE_VERSION = 'alexnet-imagenet1k-v1-early2x2-late2x2-eight-frames-v1'


def load_train_eeg(root, subject):
    path = Path(root) / 'trn' / f'sub-{subject:02d}_trn.npz'
    with np.load(path) as z:
        ids = np.array(sorted(map(int, z.files)), dtype=np.int64)
        eeg = np.stack([z[str(int(i))] for i in ids])
    return ids, downsample_eeg(eeg)


def load_test_eeg(root, subject):
    eeg = np.load(Path(root) / 'tst' / f'sub-{subject:02d}_tst.npy')
    return downsample_eeg(eeg)


def downsample_eeg(eeg):
    eeg = np.asarray(eeg)
    if eeg.ndim != 3 or eeg.shape[1:] != (17, 1500):
        raise ValueError(f'Expected EEG (N,17,1500), got {eeg.shape}')
    if not np.isfinite(eeg).all():
        raise ValueError('EEG has NaN/Inf values; inspect the original file')
    return decimate(eeg, DECIMATION, axis=-1, ftype='fir', zero_phase=True).astype(np.float32)


def read_frames(video_path, n_frames=N_FRAMES):
    """Sample fixed temporal positions; fail rather than silently discard a frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise ValueError(f'Video has no readable frame count: {video_path}')
    indices = np.rint(np.linspace(0, count - 1, n_frames)).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise IOError(f'Cannot decode frame {idx} of {video_path}')
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_video_features(video_dir, ids, cache_path, device=None):
    """Return (videos, 8, 1280) frozen AlexNet frame descriptors.

    Early-layer activation maps retain low-level structure; late-layer maps
    add higher-level information. Both are spatially pooled to 2x2 --
    no flattened raw pixels. The feature cache is shared across subjects.
    """
    import torch
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    from torchvision.models import AlexNet_Weights, alexnet

    ids = np.asarray(ids, dtype=np.int64)
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if (np.array_equal(z['ids'], ids)
                    and str(z['version'].item()) == FEATURE_VERSION
                    and z['frames'].shape == (len(ids), N_FRAMES, 1280)):
                print(f'Loaded {len(ids)} videos from {cache_path}')
                return z['frames'].astype(np.float32)
        raise ValueError(f'Incompatible cache at {cache_path}; delete and re-extract')

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weights = AlexNet_Weights.IMAGENET1K_V1
    network = alexnet(weights=weights).to(device).eval()
    early = network.features[:3]
    late = network.features[3:]
    preprocess = weights.transforms()
    features = np.empty((len(ids), N_FRAMES, 1280), dtype=np.float32)
    video_dir = Path(video_dir)

    with torch.inference_mode():
        for i, vid in enumerate(ids):
            filename = video_dir / f'{int(vid):04d}.mp4'
            batch = torch.stack([preprocess(Image.fromarray(f))
                                 for f in read_frames(filename)]).to(device)
            h = early(batch)
            low = F.adaptive_avg_pool2d(h, (2, 2)).flatten(1)     # 64*2*2
            h = late(h)
            high = F.adaptive_avg_pool2d(h, (2, 2)).flatten(1)    # 256*2*2
            features[i] = torch.cat((low, high), 1).cpu().numpy()
            if (i + 1) % 100 == 0:
                print(f'Extracted {i + 1}/{len(ids)} from {video_dir}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=ids, frames=features.astype(np.float16),
                        version=np.array(FEATURE_VERSION),
                        torchvision_version=np.array(torchvision.__version__),
                        pretrained_weights=np.array(WEIGHTS_NAME))
    # Match the cached representation even on this first extraction run.
    return features.astype(np.float16).astype(np.float32)


def select_features(all_frames, ids):
    """Align video IDs explicitly: IDs are not array row indices."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0 or np.any(ids < 1) or np.any(ids > 1102):
        raise ValueError('Invalid video IDs')
    if all_frames.shape[0] == 1000:
        if np.any(ids > 1000):
            raise ValueError('Training cache cannot contain test IDs')
        return all_frames[ids - 1]
    if all_frames.shape[0] == 102:
        if np.any(ids < 1001):
            raise ValueError('Test cache cannot contain training IDs')
        return all_frames[ids - 1001]
    raise ValueError('Expected complete training (1000) or test (102) cache')


def video_matrix(frame_features, mode):
    """Turn eight AlexNet frame embeddings into one video description."""
    frame_features = np.asarray(frame_features, dtype=np.float32)
    if frame_features.ndim != 3 or frame_features.shape[1:] != (N_FRAMES, 1280):
        raise ValueError(f'Bad frame feature shape: {frame_features.shape}')
    if mode == 'mean':
        return frame_features.mean(axis=1)
    if mode == 'halves':
        # Keep the beginning and end of the clip separate. This is still
        # frame-based; we never use flattened raw image pixels.
        first_half = frame_features[:, :4].mean(axis=1)
        second_half = frame_features[:, 4:].mean(axis=1)
        return np.concatenate([first_half, second_half], axis=1)
    if mode == 'frames':
        return frame_features.reshape(len(frame_features), -1)
    raise ValueError(f'Unknown feature mode: {mode}')


def fit_video_projection(frames, mode, n_components=64):
    """Fit feature scaling and PCA using training videos ONLY."""
    X = video_matrix(frames, mode)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n_components = min(n_components, len(Xs) - 1, Xs.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized',
              iterated_power=3, random_state=42).fit(Xs)
    pc_scaler = StandardScaler().fit(pca.transform(Xs))
    projection = dict(mode=mode, scaler=scaler, pca=pca, pc_scaler=pc_scaler)
    return projection, project_video(projection, frames)


def project_video(projection, frames):
    X = video_matrix(frames, projection['mode'])
    return projection['pc_scaler'].transform(
        projection['pca'].transform(projection['scaler'].transform(X)))


def eeg_window(eeg, window):
    a, b = (round(float(s) * EEG_HZ) for s in window)
    if not 0 <= a < b <= 300:
        raise ValueError(f'Invalid EEG time window: {window}')
    return eeg[:, :, a:b]


def fit_eeg_encoder(projected_train, eeg_train, window, projection):
    """RidgeCV maps frame embeddings to early EEG time samples."""
    target = eeg_window(eeg_train, window)
    channel_mean = target.mean(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(target.std(axis=(0, 2), keepdims=True), 1e-6)
    Y = ((target - channel_mean) / channel_std).reshape(len(target), -1)
    ridge = RidgeCV(alphas=np.logspace(-1, 5, 10)).fit(projected_train, Y)
    return dict(projection=projection, window=tuple(window), ridge=ridge,
                channel_mean=channel_mean, channel_std=channel_std,
                eeg_shape=target.shape[1:], n_train=len(target),
                weights=WEIGHTS_NAME)


def similarity(encoder, test_eeg, test_frames, center_trials=False):
    """Rank candidates using predicted and observed EEG patterns.

    ``center_trials`` removes each EEG feature's average across the 102
    recordings, and the corresponding average across the 102 video candidates.
    This is unsupervised centering: we never use labels, fit a model to test
    recordings, or assume each video appears exactly once.

    The original order of the test EEG rows is never changed.
    """
    projected = project_video(encoder['projection'], test_frames)
    predicted = encoder['ridge'].predict(projected).astype(np.float32)
    recorded = eeg_window(test_eeg, encoder['window'])
    recorded = ((recorded - encoder['channel_mean']) /
                encoder['channel_std']).reshape(len(recorded), -1)

    if center_trials:
        # Every trial includes shared responses (video onset, common EEG
        # background). Remove that shared part to emphasize differences.
        # This is permitted unsupervised normalization, not training.
        if len(recorded) < 2 or len(predicted) < 2:
            raise ValueError('Trial-centering requires at least two recordings')
        recorded = recorded - recorded.mean(axis=0, keepdims=True)
        predicted = predicted - predicted.mean(axis=0, keepdims=True)

    # Per-recording correlation: no fit or optimization uses test recordings.
    def normalize(rows):
        rows = np.asarray(rows, dtype=np.float32)
        rows = rows - rows.mean(axis=1, keepdims=True)
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)

    return normalize(recorded) @ normalize(predicted).T


def ranked_video_ids(scores, candidate_ids=TEST_IDS):
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if scores.shape[1] != len(candidate_ids):
        raise ValueError('Scores/candidate count mismatch')
    return candidate_ids[np.argsort(-scores, axis=1, kind='stable')]


def validate_rankings(ranked, ground_truth):
    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    ranks = (ranked == ground_truth[:, None]).argmax(axis=1) + 1
    if not np.all((ranked == ground_truth[:, None]).sum(axis=1) == 1):
        raise AssertionError('Each ground-truth video must occur exactly once per row')
    return dict(top1=float((ranks == 1).mean()),
                top5=float((ranks <= 5).mean()),
                median_rank=float(np.median(ranks)))


def assert_submission(ranked):
    if ranked.shape != (102, 102) or not np.issubdtype(ranked.dtype, np.integer):
        raise AssertionError(f'Expected integer (102, 102), got {ranked.shape}, {ranked.dtype}')
    if not np.all(np.sort(ranked, axis=1) == TEST_IDS):
        raise AssertionError('Every row must be a permutation of VIDEO IDs 1001..1102')


def save_encoder(encoder, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(encoder, path, compress=3)
