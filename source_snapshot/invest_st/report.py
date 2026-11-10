# -*- coding: utf-8 -*-
"""report 模块（invest_st I1）：audit.json + canonical_metrics.csv + manifest.json 输出。"""
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_canonical_csv(path, rows, fieldnames):
    with open(path, 'x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_audit_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1))


def write_manifest(out_dir, inputs, artifacts, extra=None):
    """manifest：schema version、代码/输入 hash、环境、exit codes、时间。"""
    env = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': __import__('numpy').__version__,
        'pandas': __import__('pandas').__version__,
    }
    man = {
        'document': 'invest audit manifest（I1 MVP）',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'schema_version': '0.1.0',
        'inputs': {k: (str(v), sha256_file(v)) for k, v in inputs.items() if Path(v).exists()},
        'artifacts': {k: (str(v), sha256_file(v)) for k, v in artifacts.items() if Path(v).exists()},
        'environment': env,
        'exit_code': 0,
    }
    if extra:
        man.update(extra)
    p = out_dir / 'manifest.json'
    with p.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(man, ensure_ascii=False, indent=1))
    return p
