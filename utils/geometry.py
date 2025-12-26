import torch
import numpy as np

def get_geometry_stats(model, loader, device, num_classes):
    """
    마지막 피쳐층의 기하학적 구조(Intra-class, Inter-class, Anisotropy, Effective Rank 등)를 분석합니다.
    """
    model.eval()
    
    # 1. Feature 및 Label 추출
    features_list = []
    labels_list = []
    
    # 모델의 Linear layer 가중치 가져오기 (NC3 측정용 - ResNet/WideResNet/VGG 대응)
    if hasattr(model, 'module'):
        # DataParallel 사용 시
        if hasattr(model.module, 'fc'): classifier_weights = model.module.fc.weight.data
        elif hasattr(model.module, 'linear'): classifier_weights = model.module.linear.weight.data
        else: classifier_weights = None
    else:
        if hasattr(model, 'fc'): classifier_weights = model.fc.weight.data
        elif hasattr(model, 'linear'): classifier_weights = model.linear.weight.data
        else: classifier_weights = None
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            _ = model(data) # Forward pass를 통해 self.feature 업데이트
            
            # 모델 내부 feature 가져오기
            if hasattr(model, 'module'): features = model.module.feature
            else: features = model.feature
            
            features_list.append(features.cpu())
            labels_list.append(target.cpu())

    features = torch.cat(features_list).to(device) # (N, D)
    labels = torch.cat(labels_list).to(device)     # (N,)
    
    N, D = features.shape
    
    # 2. 클래스별 통계량 계산
    class_means = []
    within_class_scatter = 0
    anisotropies = []
    
    for c in range(num_classes):
        idxs = (labels == c)
        if idxs.sum() == 0: continue
            
        feats_c = features[idxs]
        mu_c = feats_c.mean(dim=0)
        class_means.append(mu_c)
        
        # 공분산 및 Trace 계산
        centered = feats_c - mu_c
        cov_c = torch.matmul(centered.T, centered) / (feats_c.shape[0] - 1)
        
        trace_c = torch.trace(cov_c)
        within_class_scatter += trace_c
        
        # Anisotropy (Lambda_max / Trace)
        eigvals = torch.linalg.eigvalsh(cov_c)
        lambda_max = eigvals[-1]
        anisotropy_c = lambda_max / trace_c if trace_c > 1e-8 else torch.tensor(0.0, device=device)
        anisotropies.append(anisotropy_c)

    class_means = torch.stack(class_means) # (K, D)
    
    # [지표 1] Within-class variance (Mean Trace)
    mean_within_class_var = (within_class_scatter / num_classes).item()
    
    # [지표 2] Inter-class mean distance
    dist_matrix = torch.cdist(class_means, class_means, p=2)
    sum_dist = dist_matrix.sum()
    n_pairs = num_classes * (num_classes - 1)
    mean_inter_class_dist = (sum_dist / n_pairs).item()
    
    # [지표 3] Anisotropy Mean
    mean_anisotropy = torch.tensor(anisotropies).mean().item()
    
    # [지표 4] Effective Rank
    global_mean = features.mean(dim=0)
    features_centered = features - global_mean
    total_cov = torch.matmul(features_centered.T, features_centered) / (N - 1)
    total_eigvals = torch.linalg.eigvalsh(total_cov)
    total_eigvals = total_eigvals[total_eigvals > 1e-6] # 노이즈 제거
    
    if len(total_eigvals) > 0:
        p_i = total_eigvals / total_eigvals.sum()
        entropy = -torch.sum(p_i * torch.log(p_i + 1e-12))
        effective_rank = torch.exp(entropy).item()
    else:
        effective_rank = 0.0

    # [추가 지표] NC2 (Cosine Sim of Class Means)
    means_norm = class_means / class_means.norm(dim=1, keepdim=True)
    cosine_sim_matrix = torch.mm(means_norm, means_norm.T)
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    mean_cosine_sim = cosine_sim_matrix[mask].mean().item()

    # [추가 지표] NC3 (Feature-Weight Alignment)
    mean_feature_weight_alignment = 0.0
    if classifier_weights is not None:
        W = classifier_weights.to(device)
        W_norm = W / W.norm(dim=1, keepdim=True)
        # 같은 클래스끼리의 코사인 유사도 평균
        alignment_scores = torch.diag(torch.mm(means_norm, W_norm.T))
        mean_feature_weight_alignment = alignment_scores.mean().item()

    return {
        "Within-class Variance": mean_within_class_var,
        "Inter-class Distance": mean_inter_class_dist,
        "Anisotropy": mean_anisotropy,
        "Effective Rank": effective_rank,
        "NC2 (Mean Sim)": mean_cosine_sim,
        "NC3 (Alignment)": mean_feature_weight_alignment
    }