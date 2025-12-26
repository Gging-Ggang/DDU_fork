"""
Script to evaluate a single model with explicit path definition.
Verified fixes:
1. Removed DataParallel (Fixes feature extraction).
2. Corrected Score Signs (Energy, Mahalanobis, kNN).
3. Added DDU (GMM) Evaluation.
4. Added NumpyEncoder for JSON serialization.
5. Includes SN Hook Metadata Patch.
"""
import os
import json
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score

# 1. Import Dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.ood_detection.mnist_ood as mnist_ood
import data.ood_detection.tiny_imagenet as tiny_imagenet

# 2. Import Networks and SN Hook for Patching
from net.resnet import resnet50, resnet18
from net.wide_resnet import wrn
from net.vgg import vgg16
import net.spectral_normalization.spectral_norm_conv_inplace as sn_lib

# =============================================================================
# [Monkey Patch] Fix KeyError: 'weight' in SpectralNormConvLoadStateDictPreHook
# =============================================================================
original_load_hook = sn_lib.SpectralNormConvLoadStateDictPreHook.__call__

def patched_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    fn = self.fn
    version = local_metadata.get("spectral_norm_conv", {}).get(fn.name + ".version", None)
    if (version is None or version < 1) and (prefix + fn.name) not in state_dict:
        if (prefix + fn.name + "_orig") in state_dict:
            return
    return original_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

sn_lib.SpectralNormConvLoadStateDictPreHook.__call__ = patched_load_hook
# =============================================================================

# 3. Import Metrics & Utils
from metrics.classification_metrics import test_classification_net, get_logits_labels
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp, confidence
from metrics.ood_metrics import get_roc_auc

from utils.geometry import get_geometry_stats
from utils.gmm_utils import get_embeddings, gmm_fit, gmm_get_logits
from utils.ood_scores import get_energy_score, MahalanobisScorer, KNNScorer
from utils.temperature_scaling import ModelWithTemperature

# =============================================================================
# [JSON Encoder] Fix 'float32 is not JSON serializable'
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# =============================================================================

# Configuration Maps
dataset_loader = {
    "cifar10": cifar10, "cifar100": cifar100, "svhn": svhn,
    "mnist": mnist_ood, "tiny_imagenet": tiny_imagenet
}
dataset_num_classes = {
    "cifar10": 10, "cifar100": 100, "svhn": 10, "mnist": 10, "tiny_imagenet": 200
}
models = {
    "resnet50": resnet50, "resnet18": resnet18, 
    "wide_resnet": wrn, "vgg16": vgg16
}
model_to_num_dim = {
    "resnet50": 2048, "resnet18": 512, 
    "wide_resnet": 640, "vgg16": 512
}

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a single model", allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Full path to the .model file")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=list(dataset_loader.keys()))
    parser.add_argument("--ood_dataset", type=str, default="svhn", choices=list(dataset_loader.keys()))
    parser.add_argument("--model", type=str, default="wide_resnet", choices=list(models.keys()))
    parser.add_argument("--sn", action="store_true", help="Spectral Normalization")
    parser.add_argument("--mod", action="store_true", help="Feature Modification")
    parser.add_argument("--coeff", type=float, default=3.0, help="Coeff for mod")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for softmax")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".", help="Save directory")
    return parser.parse_args()

def compute_auroc(id_scores, ood_scores):
    # id_scores: Higher is ID
    # ood_scores: Lower is OOD (Higher is ID)
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_scores)

def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    print(f"Target Checkpoint: {args.checkpoint_path}")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"File NOT found: {args.checkpoint_path}")

    # 1. Load Data
    num_classes = dataset_num_classes[args.dataset]
    print(f"Dataset: ID={args.dataset}, OOD={args.ood_dataset}")
    
    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=cuda)
    
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.batch_size, augment=False, val_seed=args.seed, val_size=0.1, pin_memory=cuda
    )

    if args.ood_dataset in ["mnist", "tiny_imagenet"]:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, root="./data")
    else:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=cuda)

    # 2. Build Model
    print(f"Building Model: {args.model} (SN={args.sn}, Mod={args.mod}, Coeff={args.coeff})")
    net = models[args.model](
        spectral_normalization=args.sn, 
        mod=args.mod, 
        coeff=args.coeff, 
        num_classes=num_classes, 
        temp=args.temp
    )
    
    # 3. Load Weights
    print("Loading Weights...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    
    net.to(device)
    net.eval()

    # 4. Standard Metrics
    print("\n--- Standard Metrics ---")
    (conf_matrix, accuracy, labels_list, predictions, confidences) = test_classification_net(net, test_loader, device)
    ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
    print(f"Accuracy: {accuracy:.4f}, ECE: {ece:.4f}")

    # Temperature Scaling
    t_model = ModelWithTemperature(net)
    t_model.set_temperature(val_loader)
    (t_conf_matrix, t_accuracy, t_labels_list, t_predictions, t_confidences) = test_classification_net(t_model, test_loader, device)
    t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)
    print(f"Scaled ECE: {t_ece:.4f} (Optimal Temp: {t_model.temperature:.4f})")

    # 5. Extract Features for OOD
    print("\n--- Extracting Features ---")
    dim = model_to_num_dim[args.model]
    
    # Using CPU to avoid OOM during feature collection
    train_feats, train_lbls = get_embeddings(net, train_loader, num_dim=dim, dtype=torch.float, device=device, storage_device=torch.device('cpu'))
    test_feats, _ = get_embeddings(net, test_loader, num_dim=dim, dtype=torch.float, device=device, storage_device=torch.device('cpu'))
    ood_feats, _ = get_embeddings(net, ood_test_loader, num_dim=dim, dtype=torch.float, device=device, storage_device=torch.device('cpu'))
    
    test_logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    results = {
        "accuracy": accuracy,
        "ece": ece,
        "t_ece": t_ece,
        "metrics": {}
    }

    # 6. Compute OOD Scores
    print("\n--- Computing OOD Scores ---")
    
    # (1) MSP
    (_, _, _), (_, _, _), msp_auc, _ = get_roc_auc(net, test_loader, ood_test_loader, confidence, device, confidence=True)
    results["metrics"]["msp_auroc"] = msp_auc
    print(f"MSP AUROC: {msp_auc:.4f}")

    # (2) Entropy
    (_, _, _), (_, _, _), ent_auc, _ = get_roc_auc(net, test_loader, ood_test_loader, entropy, device)
    results["metrics"]["entropy_auroc"] = ent_auc
    print(f"Entropy AUROC: {ent_auc:.4f}")

    # (3) Energy Score
    id_energy = get_energy_score(test_logits, T=1.0)
    ood_energy = get_energy_score(ood_logits, T=1.0)
    energy_auc = compute_auroc(id_energy.cpu().numpy(), ood_energy.cpu().numpy())
    results["metrics"]["energy_auroc"] = energy_auc
    print(f"Energy AUROC: {energy_auc:.4f} (T=1.0)")

    # (4) DDU (GMM)
    print("Fitting GMM (DDU)...")
    try:
        # Fit GMM on Train features
        gmm_model, _ = gmm_fit(train_feats, train_lbls, num_classes)
        
        # Calculate Log-Likelihoods (Density)
        # GMM log-prob is higher for ID data.
        ddu_id_logits = gmm_get_logits(gmm_model, test_feats)
        ddu_ood_logits = gmm_get_logits(gmm_model, ood_feats)
        
        # Marginal Log-Likelihood p(x) = log(sum(exp(log_p(x|c) + log_p(c))))
        # Assuming uniform prior p(c), this is proportional to logsumexp of class conditional log-probs.
        ddu_id_score = torch.logsumexp(ddu_id_logits, dim=1)
        ddu_ood_score = torch.logsumexp(ddu_ood_logits, dim=1)
        
        ddu_auc = compute_auroc(ddu_id_score.numpy(), ddu_ood_score.numpy())
        results["metrics"]["ddu_auroc"] = ddu_auc
        print(f"DDU (GMM) AUROC: {ddu_auc:.4f}")
    except Exception as e:
        print(f"Error computing DDU: {e}")
        results["metrics"]["ddu_auroc"] = 0.0

    # (5) Mahalanobis
    maha = MahalanobisScorer()
    maha.fit(train_feats, train_lbls, num_classes)
    maha_id = maha.score(test_feats)
    maha_ood = maha.score(ood_feats)
    maha_auc = compute_auroc(maha_id.numpy(), maha_ood.numpy())
    results["metrics"]["maha_auroc"] = maha_auc
    print(f"Mahalanobis AUROC: {maha_auc:.4f}")

    # (6) kNN
    knn = KNNScorer(k=50)
    knn.fit(train_feats)
    knn_id = knn.score(test_feats)
    knn_ood = knn.score(ood_feats)
    knn_auc = compute_auroc(knn_id.numpy(), knn_ood.numpy())
    results["metrics"]["knn_auroc"] = knn_auc
    print(f"kNN AUROC: {knn_auc:.4f}")

    # 7. Geometry Analysis
    print("\n--- Analyzing Geometry ---")
    try:
        geo_stats = get_geometry_stats(net, train_loader, device, num_classes)
        results["geometry"] = geo_stats
        print(geo_stats)
    except Exception as e:
        print(f"Error computing geometry stats: {e}")

    # 8. Save
    ckpt_name = os.path.basename(args.checkpoint_path)
    save_name = f"res_{ckpt_name}_{args.ood_dataset}.json"
    save_path = os.path.join(args.output_dir, save_name)
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"\nSaved to: {save_path}")

if __name__ == "__main__":
    main()