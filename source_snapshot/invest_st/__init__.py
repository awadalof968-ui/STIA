# -*- coding: utf-8 -*-
"""invest-st — INVariance-aware Evaluation of Spatial Transcriptomics.

供体感知、设计感知的多样本空间转录组整合审计工作流：
- provenance:     M1 manifest / 输入哈希 / provenance completeness
- validate:       元数据完整性 / spot_id 唯一性 / latent↔metadata 键连接对齐 / 嵌套结构
- identifiability: batch×condition 设计可识别性判定（IDENTIFIABLE / PARTIALLY / NOT）
- metrics:        三轴指标封装（conditional batch: iLISI/ASW/decoder/graph；disease；spatial）
- fairness:       M4 特征面板 / 表示公平性 / marker 泄漏检查
- statistics:     M5 统计单位 / 多重校正 / 伪重复 / CI 覆盖 / 功效
- claims:         M8 claim 治理（allowed/forbidden/uncertain + 证据绑定）
- decision:       M9 非补偿硬门与最终决策
- report:         audit.json + canonical_metrics.csv + manifest.json
"""
__version__ = "1.3.8"
__all__ = ["provenance", "validate", "identifiability", "metrics", "fairness",
           "statistics", "claims", "decision", "report", "config"]
