# -*- coding: utf-8 -*-
"""invest_st.metrics — 三轴指标封装 + 顺序不变性审计（I1 MVP + D22b OSI 门）。

复用项目内已验证的纯 numpy 实现（_impl/ 内自包含副本），
避免重复实现与结果漂移；不依赖任何项目脚本或数据路径。
"""
from ._impl._x3_metrics import (  # noqa: F401
    batch_decoding_accuracy,
    ilisi_np,
    kmeans_np,
    ari_np,
    nmi_np,
    spatial_ari,
)
from ._impl._x3_stageC_bc1_v6_metrics import (  # noqa: F401
    graph_connectivity,
    cond_batch_decoder,
    cond_graph_connectivity,
    cond_metric_donor,
)


def batch_axis_metrics(latent, batch, layer, group, donor,
                       k_ilisi=90, k_graph=20, max_per_stratum=1500):
    """返回条件批次轴四指标（spot_weighted + equal_stratum + strata 明细）。"""
    from ._impl._x3_stageC_bc1_v6_metrics import (
        cond_batch_decoder,
        cond_graph_connectivity,
    )
    from ._impl._x3_stageC_bc1_v5 import cond_ilisi_donor, cond_asw
    res = {
        'cond_ilisi': cond_ilisi_donor(latent, batch, layer, group, donor, max_per_stratum)[0],
        'cond_asw': cond_asw(latent, batch, layer, group, max_per_stratum)[0],
        'cond_decoder': cond_batch_decoder(latent, batch, layer, group, donor, max_per_stratum),
        'cond_graph': cond_graph_connectivity(latent, batch, layer, group, donor, k=k_graph, max_per_stratum=max_per_stratum),
    }
    return res


def disease_axis_metrics(embedding, disease_labels, donor_labels, seed=0, n_folds=5):
    """供体级疾病信号保留（占位实现：donor-level CV 分类 AUC）。
    I1 MVP 提供骨架；论文级实现以 canonical 表中 disease_score 为准。"""
    # 供体级交叉验证：每个 donor fold，训练分类器（表达均值特征）预测疾病
    # 返回 macro-AUC；donor 数过少时返回 None
    return None


def spatial_axis_metrics(latent, layer_labels, k=8, seed=0):
    """空间结构保留：kmeans(k) 聚类 vs 层标签 的 ARI/NMI。"""
    km = kmeans_np(latent, k, seed=seed)
    return {'spatial_ari': ari_np(layer_labels, km),
            'spatial_nmi': nmi_np(layer_labels, km)}
