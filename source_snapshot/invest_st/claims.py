# -*- coding: utf-8 -*-
"""STIA M8: claim governance——把论文表述分类为 allowed/forbidden/uncertain 并绑定证据。

对应计划: MULTIDIMENSIONAL_AUDIT_TOOL_AND_MULTIMODEL_VALIDATION_PLAN T1 (M8)
实现: 确定性禁词/模式检测（D22d forbidden_claims 词汇）+ 证据绑定检查。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# D22d/D22_PAPER_POSITIONING 冻结的 forbidden claim 模式（论文主边界）
FORBIDDEN_PATTERNS = [
    # 顺序无关 disease retention
    (r'order[- ]independent', 'order_independent_claim'),
    (r'regardless of (input )?order', 'order_independent_claim'),
    (r'universal(ly)? (superior|better)', 'universal_superiority'),
    (r'superior to (all|every|GraphST|PCA|Harmony)', 'universal_superiority'),
    (r'confirmatory external (validation|replication)', 'confirmatory_external_claim'),
    (r'independently validated', 'confirmatory_external_claim'),
    (r'clinical (diagnos|utility|decision)', 'clinical_claim'),
    (r'first (to )?(discover|show|demonstrate)', 'first_discovery_claim'),
    (r'prove[sd]? (that )?(shared|gamma)', 'mechanism_causality_claim'),
    (r'causal(ly)? (source|driver)', 'mechanism_causality_claim'),
    (r'fix(es|ed)? (by|with) (re)?order', 'order_repair_claim'),
    (r'repaired (method|model)', 'repaired_method_claim'),
]

# 需要证据身份的关键词（触发 evidence 绑定检查）
EVIDENCE_REQUIRED_PATTERNS = [
    r'OSI', r'order', r'shared', r'gamma', r'latent', r'recall', r'FPR',
    r'false[- ]positive', r'blocker', r'provenance', r'forbidden claim',
]


@dataclass
class ClaimCheck:
    claim_text: str
    forbidden_hits: List[Dict] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    evidence_bound: bool = False
    decision: str = 'UNCERTAIN'  # ALLOWED / FORBIDDEN / UNCERTAIN

    def to_dict(self) -> Dict:
        return {
            'claim_text': self.claim_text,
            'forbidden_hits': self.forbidden_hits,
            'evidence_required': self.evidence_required,
            'evidence_bound': self.evidence_bound,
            'decision': self.decision,
        }


def audit_claim(claim_text: str, evidence_ids: Optional[List[str]] = None,
                evidence_inventory: Optional[Dict] = None) -> ClaimCheck:
    """对单条 claim 做治理检查。

    evidence_ids: 该 claim 引用的证据 ID 列表。
    evidence_inventory: {evidence_id: {status, artifact, ...}}。
    """
    cc = ClaimCheck(claim_text=claim_text)
    evidence_ids = evidence_ids or []

    # 1. forbidden 模式检测
    for pat, category in FORBIDDEN_PATTERNS:
        if re.search(pat, claim_text, re.IGNORECASE):
            cc.forbidden_hits.append({'pattern': pat, 'category': category})
            cc.decision = 'FORBIDDEN'

    # 2. 证据需求检测
    for pat in EVIDENCE_REQUIRED_PATTERNS:
        if re.search(pat, claim_text, re.IGNORECASE):
            cc.evidence_required.append(pat)

    # 3. 证据绑定检查（claim 需要证据时必须引用且证据存在）
    if cc.evidence_required:
        if evidence_ids:
            # 检查引用的证据是否在 inventory 且 status 可用
            missing = [e for e in evidence_ids
                       if not (evidence_inventory and e in evidence_inventory
                               and evidence_inventory[e].get('status') in
                               ('PASS', 'CONDITIONAL_PASS', 'COMPLETE'))]
            cc.evidence_bound = not missing
            if cc.evidence_bound:
                if cc.decision == 'UNCERTAIN':
                    cc.decision = 'ALLOWED'
            elif cc.decision != 'FORBIDDEN':
                cc.decision = 'UNCERTAIN'
        else:
            cc.evidence_bound = False
            if cc.decision != 'FORBIDDEN':
                cc.decision = 'UNCERTAIN'
    else:
        cc.evidence_bound = True  # 无证据要求的 claim 不需要绑定
        if cc.decision == 'UNCERTAIN':
            cc.decision = 'ALLOWED'

    return cc


def audit_claims_batch(claims: List[Dict], evidence_inventory: Optional[Dict] = None) -> List[Dict]:
    """批量审计 claims: [{'claim': str, 'evidence': [ids]}] -> [ClaimCheck dict]。"""
    return [audit_claim(c.get('claim', ''), c.get('evidence'), evidence_inventory).to_dict()
            for c in claims]
