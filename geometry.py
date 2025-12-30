# geometry.py
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# 1) Dataset / Dataloader
# -----------------------------
def get_id_dataset(dataset_name: str, data_root: str, split: str):
    dataset_name = dataset_name.lower()
    split = split.lower()
    train = (split == "train")

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = datasets.CIFAR10(root=data_root, train=train, download=True, transform=tfm)
        num_classes = 10

    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = datasets.CIFAR100(root=data_root, train=train, download=True, transform=tfm)
        num_classes = 100

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return ds, num_classes


# -----------------------------
# 2) Model builder (YOU MUST ADAPT THIS)
# -----------------------------
def build_model(model_name: str, num_classes: int, sn: bool, coeff: float):
    """
    IMPORTANT:
    - Your checkpoint contains keys like conv1.weight_orig / weight_u / weight_v / weight_sigma.
      That means you MUST build the exact same architecture used in train.py (approx SN conv).
    - Replace the body of this function with your project's model constructor.

    Expected:
    - returned model has attribute `fc` (final linear layer)
    - model.fc.weight shape == (num_classes, feature_dim)
    """
    model_name = model_name.lower()

    # ---- EXAMPLE PLACEHOLDER (YOU MUST REPLACE) ----
    # from your_project.models import resnet18  # <-- change this
    # model = resnet18(num_classes=num_classes, sn=sn, coeff=coeff)
    # return model

    raise RuntimeError(
        "build_model() is a placeholder. Replace it with your train.py model constructor "
        "so that the checkpoint state_dict keys match."
    )


def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    # state_dict-only
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and "state_dict" not in obj:
        return obj
    # common checkpoint dict
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    raise ValueError("Unknown checkpoint format. Expected state_dict-only or dict with key 'state_dict'.")


# -----------------------------
# 3) NC metrics helpers
# -----------------------------
def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x, p="fro")


def make_etf_gram(K: int, device, dtype) -> torch.Tensor:
    I = torch.eye(K, device=device, dtype=dtype)
    ones = torch.ones((K, K), device=device, dtype=dtype)
    return (I - ones / K) / math.sqrt(K - 1)


def pairwise_mean_dist(M: torch.Tensor) -> torch.Tensor:
    # M: (K, d)
    K = M.shape[0]
    D = torch.cdist(M, M, p=2)
    idx = torch.triu_indices(K, K, offset=1)
    return D[idx[0], idx[1]].mean()


def effective_rank_from_cov(cov: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # cov: (d, d) symmetric PSD
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    s = evals.sum()
    if s < eps:
        return torch.tensor(0.0, device=cov.device, dtype=cov.dtype)
    p = evals / (s + eps)
    ent = -(p * torch.log(p + eps)).sum()
    return torch.exp(ent)


def anisotropy_lambda1_over_trace(cov: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    tr = evals.sum()
    if tr < eps:
        return torch.tensor(0.0, device=cov.device, dtype=cov.dtype)
    return evals[-1] / (tr + eps)


# -----------------------------
# 4) Main computation
# -----------------------------
@torch.no_grad()
def compute_geometry_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    eig_device: torch.device,
    dtype: torch.dtype = torch.float64,
    save_matrices: bool = False,
    matrices_out: str = None,
):
    model.eval()
    model.to(device)

    # Hook to capture penultimate features (input to fc)
    feats_buf = {}
    def fc_prehook(module, inputs):
        feats_buf["h"] = inputs[0].detach()

    handle = model.fc.register_forward_pre_hook(fc_prehook)

    # We accumulate:
    # - n_c
    # - sum_h_c: (K,d)
    # - sum_hhT_c: (K,d,d)  (float64)
    # - sum_h_global: (d,)
    K = num_classes
    n_c = torch.zeros(K, dtype=torch.long)
    sum_h_c = None            # init after we know d
    sum_h_global = None
    sum_hhT_c = None

    N_total = 0
    feature_dim = None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _ = model(images)  # forward triggers hook
        h = feats_buf.pop("h")  # (B,d)

        if feature_dim is None:
            feature_dim = h.shape[1]
            d = feature_dim
            sum_h_c = torch.zeros((K, d), dtype=dtype)
            sum_h_global = torch.zeros((d,), dtype=dtype)
            sum_hhT_c = torch.zeros((K, d, d), dtype=dtype)

        h64 = h.to(dtype=dtype).detach().cpu()  # accumulate on CPU (stable, avoids GPU mem spikes)
        y = labels.detach().cpu()

        B = h64.shape[0]
        N_total += B

        sum_h_global += h64.sum(dim=0)

        # per-class accumulation
        for c in torch.unique(y):
            c_int = int(c.item())
            idx = (y == c_int)
            X = h64[idx]  # (n,d)
            nc = X.shape[0]
            if nc == 0:
                continue
            n_c[c_int] += nc
            sum_h_c[c_int] += X.sum(dim=0)
            sum_hhT_c[c_int] += X.t().mm(X)  # (d,d)

    handle.remove()

    if feature_dim is None:
        raise RuntimeError("No data passed through loader; feature_dim is None.")

    # Means
    N = int(N_total)
    mu_G = sum_h_global / max(N, 1)

    mu_c = torch.zeros_like(sum_h_c)
    for c in range(K):
        if n_c[c] > 0:
            mu_c[c] = sum_h_c[c] / n_c[c].item()

    # (a) within-class scatter Sigma_W
    # Sigma_W = (1/N) * sum_c ( sum_hhT_c - n_c * mu_c mu_c^T )
    d = feature_dim
    SW = torch.zeros((d, d), dtype=dtype)
    for c in range(K):
        nc = n_c[c].item()
        if nc == 0:
            continue
        mc = mu_c[c].view(d, 1)
        SW += (sum_hhT_c[c] - nc * (mc @ mc.t()))
    Sigma_W = SW / max(N, 1)

    # (b) between-class scatter Sigma_B (class-means scatter)
    SB = torch.zeros((d, d), dtype=dtype)
    for c in range(K):
        dc = (mu_c[c] - mu_G).view(d, 1)
        SB += dc @ dc.t()
    Sigma_B = SB / K

    # (c) NC1
    SigmaB_pinv = torch.linalg.pinv(Sigma_B)
    nc1 = (torch.trace(Sigma_W @ SigmaB_pinv) / K)

    # (d) NC2 (ETF of W)
    W = model.fc.weight.detach().to(dtype=dtype).cpu()  # (K,d)
    WWt = W @ W.t()
    G_W = WWt / (fro_norm(WWt) + 1e-12)
    G_ETF = make_etf_gram(K, device=G_W.device, dtype=dtype)
    nc2 = fro_norm(G_W - G_ETF)

    # (e) NC3 (relation between W and H)
    H = (mu_c - mu_G).t().contiguous()  # (d,K)
    WH = W @ H  # (K,K)
    G_WH = WH / (fro_norm(WH) + 1e-12)
    nc3 = fro_norm(G_WH - G_ETF)

    # (f) inter-class mean distance
    inter_class_mean_dist = pairwise_mean_dist(mu_c)

    # (g) anisotropy_mean (class-wise lambda1/trace)
    # (h) eff_rank (class-wise effective rank)
    anis_list = []
    effrank_list = []

    # eigen computations can be moved to eig_device if you want, but we already have cov on CPU here.
    # We'll compute on CPU by default for stability; set eig_device to cuda if you want faster.
    for c in range(K):
        nc = n_c[c].item()
        if nc <= 1:
            continue
        mc = mu_c[c].view(d, 1)
        cov_c = (sum_hhT_c[c] / nc) - (mc @ mc.t())  # (d,d)
        cov_c = 0.5 * (cov_c + cov_c.t())  # symmetrize

        if eig_device.type == "cuda":
            cov_c_dev = cov_c.to(eig_device)
            anis = anisotropy_lambda1_over_trace(cov_c_dev).cpu()
            er = effective_rank_from_cov(cov_c_dev).cpu()
        else:
            anis = anisotropy_lambda1_over_trace(cov_c)
            er = effective_rank_from_cov(cov_c)

        anis_list.append(anis.item())
        effrank_list.append(er.item())

    anisotropy_mean = float(np.mean(anis_list)) if len(anis_list) > 0 else 0.0
    eff_rank = float(np.mean(effrank_list)) if len(effrank_list) > 0 else 0.0

    # Summaries for (a),(b) to store in JSON
    within_trace = torch.trace(Sigma_W).item()
    within_fro = fro_norm(Sigma_W).item()
    between_trace = torch.trace(Sigma_B).item()
    between_fro = fro_norm(Sigma_B).item()

    out = {
        "N": N,
        "K": K,
        "feature_dim": d,

        # (a) within-class scatter (summary)
        "within_scatter_trace": within_trace,
        "within_scatter_fro": within_fro,

        # (b) between-class scatter (summary)
        "between_scatter_trace": between_trace,
        "between_scatter_fro": between_fro,

        # (c)~(e) NC metrics
        "nc1": float(nc1.item()),
        "nc2": float(nc2.item()),
        "nc3": float(nc3.item()),

        # (f)
        "inter_class_mean_dist": float(inter_class_mean_dist.item()),

        # (g)
        "anisotropy_mean": float(anisotropy_mean),

        # (h)
        "eff_rank": float(eff_rank),

        # optional debug
        "anisotropy_per_class_count": int(len(anis_list)),
        "eff_rank_per_class_count": int(len(effrank_list)),
    }

    if save_matrices:
        if matrices_out is None:
            raise ValueError("matrices_out must be provided when save_matrices=True.")
        np.savez_compressed(
            matrices_out,
            Sigma_W=Sigma_W.numpy(),
            Sigma_B=Sigma_B.numpy(),
            mu_c=mu_c.numpy(),
            mu_G=mu_G.numpy(),
        )
        out["matrices_npz"] = str(matrices_out)

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--sn", action="store_true")
    p.add_argument("--coeff", type=float, default=3.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eig-device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--out-json", type=str, required=True)

    p.add_argument("--save-matrices", action="store_true")
    p.add_argument("--matrices-out", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    eig_device = torch.device(args.eig_device if (args.eig_device == "cuda" and torch.cuda.is_available()) else "cpu")

    ds, K = get_id_dataset(args.dataset, args.data_root, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # build model (must match train.py)
    model = build_model(args.model, num_classes=K, sn=args.sn, coeff=args.coeff)

    # load checkpoint
    sd = load_checkpoint_state_dict(args.checkpoint_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print("[Warning] load_state_dict(strict=False) mismatch")
        print("  missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    # compute metrics
    matrices_out = args.matrices_out
    if args.save_matrices and matrices_out is None:
        # default matrices path next to json
        matrices_out = str(Path(args.out_json).with_suffix(".npz"))

    metrics = compute_geometry_metrics(
        model=model,
        loader=loader,
        num_classes=K,
        device=device,
        eig_device=eig_device,
        dtype=torch.float64,
        save_matrices=args.save_matrices,
        matrices_out=matrices_out,
    )

    # add metadata
    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["dataset"] = args.dataset
    metrics["split"] = args.split
    metrics["model"] = args.model
    metrics["sn"] = bool(args.sn)
    metrics["coeff"] = float(args.coeff)
    metrics["device"] = str(device)
    metrics["eig_device"] = str(eig_device)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
