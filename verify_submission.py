"""Verify frozen files and rescore active comparisons; Python standard library only."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib
B=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
manifest=read(B/'MANIFEST.json');bad=[]
for r in manifest['files']:
 p=B/r['file']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:bad.append(r['file'])
assert not bad,('Hash mismatch',bad)
gold600=read(B/'case_records/GOLD_600.json')['cases']
with (B/'case_records/PRIMARY_CASE_RESULTS_208.csv').open(encoding='utf-8-sig',newline='') as f:
 gold208=[{'case_id':r['case_id'],'action':r['expected_action'],'families':r['expected_families'].split(';') if r['expected_families'] else []} for r in csv.DictReader(f)]
expected={(600,'STIA'):(591,585),(600,'B6'):(545,503),(208,'STIA'):(206,202),(208,'B6'):(196,168)}
results=[]
for n,gold in [(600,gold600),(208,gold208)]:
 g={r['case_id']:r for r in gold};assert len(g)==n
 for m in ['STIA','B6']:
  records=read(B/f'case_records/{m}_{n}.json')['records'];assert len(records)==n
  assert len(set(r['case_id'] for r in records))==n and {r['case_id'] for r in records}==set(g)
  action=complete=0;metrics={k:Counter() for k in ['custody','identity','design','features','statistics','stability','quality']}
  for r in records:
   truth=g[r['case_id']];a='ACCEPT' if r['decision']=='ACCEPT_WITH_SCOPE' else r['decision']
   p=set(r['predicted_families']);t=set(truth['families']);ok=a==truth['action'];action+=ok;complete+=ok and p==t
   for k,v in metrics.items():v['tp']+=k in p and k in t;v['fp']+=k in p-t;v['fn']+=k in t-p
  assert (action,complete)==expected[n,m]
  f1={k:2*v['tp']/(2*v['tp']+v['fp']+v['fn']) for k,v in metrics.items()}
  results.append({'cases':n,'system':m,'action_correct':action,'complete_correct':complete,'macro_f1':sum(f1.values())/7,'family_f1':f1})
print(json.dumps({'status':'PASS','files_verified':len(manifest['files']),'results':results},indent=2))
