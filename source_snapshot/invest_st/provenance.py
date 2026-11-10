# -*- coding: utf-8 -*-
"""STIA M1: manifest_provenance_and_input_integrity。

对应计划: MULTIDIMENSIONAL_AUDIT_TOOL T1 (M1)
功能:
1. 生成/校验 run manifest（输入、配置、代码、环境、种子身份）
2. 计算输入文件哈希
3. provenance completeness 评分（H0/H5 硬门基础）
"""
import hashlib
import json
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


def sha256_file(path: Path) -> str:
    """文件 SHA-256（分块读，大文件安全）。"""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def environment_identity() -> Dict:
    """记录环境身份（可复现性）。"""
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'hostname': socket.gethostname(),
        'pid': str(__import__('os').getpid()),
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def create_manifest(*, case_id: str, schema_version: str, input_files: Dict[str, str],
                    config: Dict, software: Dict, seeds: Dict, orders: List[str],
                    analysis_unit: str, extra: Optional[Dict] = None) -> Dict:
    """创建 run manifest。

    input_files: {描述: 路径}
    config: 配置键值（含 config_hash）
    software: {name: version}
    seeds: {data_seed: int, model_seed: int}
    orders: ['A_original', ...]
    """
    hashes = {}
    missing = []
    for name, path in input_files.items():
        p = Path(path)
        if p.exists():
            hashes[name] = {'path': str(p), 'sha256': sha256_file(p),
                            'size_bytes': p.stat().st_size}
        else:
            missing.append(name)

    manifest = {
        'case_id': case_id,
        'schema_version': schema_version,
        'run_id': f'{case_id}-{uuid.uuid4().hex[:12]}',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'input_hashes': hashes,
        'config': config,
        'software': software,
        'seeds': seeds,
        'orders': orders,
        'analysis_unit': analysis_unit,
        'environment': environment_identity(),
        'missing_inputs': missing,
        **(extra or {}),
    }
    # 自哈希（manifest 本身）
    manifest['manifest_sha256'] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return manifest


def provenance_completeness(manifest: Dict) -> Dict:
    """provenance completeness（H5：accepted cases = 1.00）。

    计分项: case_id / schema / input_hashes 齐全 / config_hash / software /
            seeds / orders / analysis_unit / 无 missing inputs。
    """
    checks = {
        'case_id': bool(manifest.get('case_id')),
        'schema_version': bool(manifest.get('schema_version')),
        'input_hashes': bool(manifest.get('input_hashes')),
        'no_missing_inputs': len(manifest.get('missing_inputs', [])) == 0,
        'config': bool(manifest.get('config')),
        'software': bool(manifest.get('software')),
        'seeds': bool(manifest.get('seeds')),
        'orders': len(manifest.get('orders', [])) >= 2,
        'analysis_unit': bool(manifest.get('analysis_unit')),
        'manifest_hash': bool(manifest.get('manifest_sha256')),
    }
    n_ok = sum(1 for v in checks.values() if v)
    completeness = n_ok / len(checks)
    return {
        'completeness': round(completeness, 4),
        'complete': completeness >= 1.0,
        'checks': checks,
        'h5_pass': completeness >= 1.0,
    }


def verify_manifest(manifest: Dict, input_files: Dict[str, str]) -> Dict:
    """验证 manifest 的输入哈希是否与当前文件一致（防篡改/漂移）。"""
    mismatches = []
    for name, path in input_files.items():
        p = Path(path)
        if not p.exists():
            mismatches.append({'name': name, 'issue': 'missing'})
            continue
        cur = sha256_file(p)
        recorded = manifest.get('input_hashes', {}).get(name, {}).get('sha256')
        if recorded and cur != recorded:
            mismatches.append({'name': name, 'issue': 'hash_mismatch'})
    return {'verified': not mismatches, 'mismatches': mismatches}
