# -*- coding: utf-8 -*-
"""STIA M9: repair_and_final_decision_safety——非补偿硬门和最终决策。

对应计划: MULTIDIMENSIONAL_AUDIT_TOOL_AND_MULTIMODEL_VALIDATION_PLAN T1 (M9)
规则:
- 任何必须模块 FAIL 或 UNRESOLVED -> 整体 FAIL / UNRESOLVED（不得高分补偿）
- NOT_APPLICABLE 模块不参与成功分母
- 输出 PASS / CONDITIONAL_PASS / FAIL / UNRESOLVED / NOT_APPLICABLE
"""
from typing import Dict, List, Optional

# 模块依赖（M1 先于一切；M9 终局）
MANDATORY_MODULES = ['M1', 'M2', 'M3', 'M5', 'M6', 'M8']  # M4/M7/M9 视场景
MODULE_PRECEDENCE = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']


def final_decision(module_results: Dict[str, str],
                   mandatory: Optional[List[str]] = None,
                   applicability: Optional[Dict[str, bool]] = None) -> Dict:
    """非补偿最终决策。

    module_results: {module_id: 'PASS'|'CONDITIONAL_PASS'|'FAIL'|'UNRESOLVED'|'NOT_APPLICABLE'}
    mandatory: 必须通过的模块（默认 MANDATORY_MODULES）
    applicability: {module_id: bool} 模块是否适用（False = NA，不参与）
    """
    mandatory = mandatory or MANDATORY_MODULES
    applicability = applicability or {}

    issues = []
    n_mandatory_applicable = 0
    n_fail = 0
    n_unresolved = 0

    for m in mandatory:
        if applicability.get(m, True) is False:
            continue  # NA 不参与
        n_mandatory_applicable += 1
        st = module_results.get(m, 'UNRESOLVED')
        if st in ('FAIL',):
            n_fail += 1
            issues.append(f'{m} FAIL')
        elif st in ('UNRESOLVED',):
            n_unresolved += 1
            issues.append(f'{m} UNRESOLVED')
        elif st not in ('PASS', 'CONDITIONAL_PASS'):
            issues.append(f'{m} unexpected status {st}')

    # 非补偿规则
    if n_fail > 0:
        decision = 'FAIL'
    elif n_unresolved > 0:
        decision = 'UNRESOLVED'
    elif n_mandatory_applicable == 0:
        decision = 'NOT_APPLICABLE'
    else:
        # 全 PASS 或 CONDITIONAL_PASS
        any_conditional = any(module_results.get(m) == 'CONDITIONAL_PASS'
                              for m in mandatory
                              if applicability.get(m, True) is not False)
        decision = 'CONDITIONAL_PASS' if any_conditional else 'PASS'

    return {
        'decision': decision,
        'module_results': module_results,
        'mandatory_applicable_count': n_mandatory_applicable,
        'fail_count': n_fail,
        'unresolved_count': n_unresolved,
        'issues': issues,
        'safety_rule': 'non-compensatory: any FAIL or UNRESOLVED blocks PASS',
    }


def repair_actions(module_results: Dict[str, str]) -> List[Dict]:
    """根据模块状态给出安全修复建议。"""
    actions = []
    REPAIRS = {
        'M1': '补全 input/config/source/environment 哈希和 provenance 链',
        'M2': '修复 spot/donor 身份错位或嵌套统计单位',
        'M3': '重新定义 estimand 或披露不可识别成分',
        'M4': '统一特征面板/表示并检查泄漏',
        'M5': '修正评价器或补充 type-I/power 校准',
        'M6': '运行顺序不变性审计（OSI）并检查 seed/component 稳健性',
        'M7': '补全三轴条件指标或披露方法不适用',
        'M8': '修正 claim 表述或绑定缺失证据',
        'M9': '按非补偿规则重做最终决策',
    }
    for m, st in module_results.items():
        if st in ('FAIL', 'UNRESOLVED'):
            actions.append({'module': m, 'status': st,
                            'recommended_repair': REPAIRS.get(m, '调查并修复')})
    return actions
