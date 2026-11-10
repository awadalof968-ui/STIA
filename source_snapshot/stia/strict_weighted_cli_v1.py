"""Opt-in strict task-weighted selector; existing audit defaults are unchanged."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .task_selection_v2 import strict_weighted_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.input);payload=source.read_bytes();data=json.loads(payload.decode('utf-8-sig'))
    for key,size in [('methods',len(data['means'])),('axes',len(data['weights']))]:
        names=data.get(key)
        if not isinstance(names,list) or len(names)!=size or any(not isinstance(x,str) or not x for x in names) or len(set(names))!=size:
            raise ValueError('Explicit unique nonempty method and axis identities required')
    r=strict_weighted_guard(data['means'],data['weights'],data.get('reference',0))
    r.update(schema_version='stia.strict_weighted.v1',input_sha256=hashlib.sha256(payload).hexdigest(),
             methods=data['methods'],axes=data['axes'],selected_method=data['methods'][r['selected']],
             scope='Training-axis strict protection; no population or future-unit guarantee')
    with Path(a.output).open('x',encoding='utf-8') as f:
        json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)

if __name__=='__main__':main()
