# -*- coding: utf-8 -*-
"""invest_st CLI（I1 MVP）：audit 子命令。

用法示例：
  python -m invest_st.cli audit --adata input.h5ad --config design.yaml \
      --embedding-dir latents/ --out audit_report/
"""
import argparse
import json
import sys
from pathlib import Path

from invest_st import config, identifiability, report, validate


def cmd_audit(args):
    cfg = config.load_config(args.config)
    obs = config.load_adata_obs(args.adata, cfg)

    # 1. validate
    vres = validate.run_validate(obs, cfg)
    print(f'[validate] status={vres["status"]}')
    for i in vres['issues']:
        print(f'  - {i}')

    # 2. identifiability
    ires = identifiability.classify(obs, cfg)
    print(f'[identifiability] class={ires["class"]}')
    for r in ires['reasons']:
        print(f'  - {r}')

    # 3. 输出目录
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'audit.json'
    manifest_path = out / 'manifest.json'
    existing = [path.name for path in (audit_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing audit evidence in {out}: {', '.join(existing)}; "
            "choose a new versioned output directory"
        )
    audit = {
        'document': 'invest audit report（I1 MVP）',
        'generated_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'config': args.config,
        'adata': args.adata,
        'validate': vres,
        'identifiability': ires,
        'metrics_note': '三轴指标需 latent 嵌入目录（--embedding-dir）；本 MVP 已实现 batch/spatial 轴计算函数，见 invest_st.metrics',
    }
    report.write_audit_json(audit_path, audit)
    report.write_manifest(out, {'config': args.config, 'adata': args.adata},
                          {'audit.json': audit_path})
    print(f'[report] {audit_path}')
    return 0


def main(argv=None):
    print("NOTICE: 'invest' is the preserved I1 legacy MVP; use 'stia' for the canonical M1-M9 audit bundle.", file=sys.stderr)
    ap = argparse.ArgumentParser(prog='invest', description='INVariance-aware Evaluation of Spatial Transcriptomics (invest-st)（I1 MVP）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('audit', help='运行完整审计（validate + identifiability + 指标 + 报告）')
    pa.add_argument('--adata', required=True, help='AnnData 路径（obs 含 spot/sample/donor/disease/batch 元数据）')
    pa.add_argument('--config', required=True, help='设计 YAML/JSON 配置')
    pa.add_argument('--embedding-dir', default=None, help='latent 嵌入目录（method×representation 文件）')
    pa.add_argument('--out', default='audit_report', help='输出目录')
    pa.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
