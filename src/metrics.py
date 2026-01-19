import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3

try:
    from torchvision.models import Inception_V3_Weights
    _INCEPTION_WEIGHTS = Inception_V3_Weights.DEFAULT
except Exception:
    _INCEPTION_WEIGHTS = None

from scipy import linalg


def _get_inception(device):
    if _INCEPTION_WEIGHTS is not None:
        model = inception_v3(weights=_INCEPTION_WEIGHTS, transform_input=False).to(device)
        mean = _INCEPTION_WEIGHTS.meta["mean"]
        std = _INCEPTION_WEIGHTS.meta["std"]
    else:
        model = inception_v3(pretrained=True, transform_input=False).to(device)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Lambda(lambda x: (x + 1) / 2.0),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, preprocess


def _get_pool_features(model, x):
    # Inception v3: use adaptive avgpool output as features
    # forward to get logits while capturing avgpool output
    features = []

    def _hook(_module, _input, output):
        features.append(output)

    handle = model.avgpool.register_forward_hook(_hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    if not features:
        raise RuntimeError("Failed to capture Inception features.")
    feat = features[0]
    return feat.view(feat.size(0), -1)


def compute_activation_stats(dataloader, device, max_batches=None):
    model, preprocess = _get_inception(device)
    activations = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            images = preprocess(images)
            feats = _get_pool_features(model, images)
            activations.append(feats.cpu().numpy())
            if max_batches is not None and (i + 1) >= max_batches:
                break

    activations = np.concatenate(activations, axis=0)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def inception_score(dataloader, device, splits=10, max_batches=None):
    model, preprocess = _get_inception(device)
    preds = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            images = preprocess(images)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            if max_batches is not None and (i + 1) >= max_batches:
                break

    preds = np.concatenate(preds, axis=0)
    if preds.shape[0] < splits:
        splits = 1

    split_scores = []
    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits):(k + 1) * (preds.shape[0] // splits)]
        if part.size == 0:
            continue
        p_y = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        split_scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

    if not split_scores:
        return 0.0, 0.0

    return float(np.mean(split_scores)), float(np.std(split_scores))
