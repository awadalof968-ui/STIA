"""Frozen descriptive paired-action and resource metrics, not universal superiority."""
ACCEPT={'ACCEPT','ACCEPT_WITH_SCOPE'}
FAMILIES=('custody','identity','design','features','statistics','stability','quality')
GOLD={'mixed_provenance':'custody','duplicate_identity':'identity','batch_condition_confounding':'design',
      'feature_panel_mismatch':'features','spot_pseudoreplication':'statistics',
      'seed_identity_corruption':'stability','order_patient_shift':'stability',
      'label_geometry_corruption':'quality','disease_centroid_collapse':'quality'}
MODULE_FAMILY=dict(zip(('M1','M2','M3','M4','M5','M6','M7'),FAMILIES))
QA_FAMILY={'provenance':'custody','identity':'identity','batch_condition_confounding':'design','features':'features','statistical_unit':'statistics'}
def normalize_action(value):
    if value in ACCEPT: return 'ACCEPT'
    if value in ('REJECT','ABSTAIN'): return value
    raise ValueError('Unrecognized benchmark action')
def summarize_actions(rows):
    faults=[r for r in rows if r['truth']=='fault']; neutral=[r for r in rows if r['truth']=='neutral']
    if not faults or not neutral: raise ValueError('Both fault and neutral cases required')
    actions=[normalize_action(r['decision']) for r in rows]
    return {'cases':len(rows),'fault_total':len(faults),'neutral_total':len(neutral),
            'fault_rejected':sum(normalize_action(r['decision'])=='REJECT' for r in faults),
            'unsafe_fault_accepted':sum(normalize_action(r['decision'])=='ACCEPT' for r in faults),
            'neutral_rejected':sum(normalize_action(r['decision'])=='REJECT' for r in neutral),
            'neutral_accepted':sum(normalize_action(r['decision'])=='ACCEPT' for r in neutral),
            'abstained':actions.count('ABSTAIN')}
def localization_scores(rows,families=FAMILIES):
    if not rows: raise ValueError('No cases')
    detail={}; vocabulary=set(families)
    for r in rows:
        if not (set(r['expected'])|set(r['predicted']))<=vocabulary: raise ValueError('Unknown localization family')
    for f in families:
        tp=sum(f in r['expected'] and f in r['predicted'] for r in rows)
        fp=sum(f not in r['expected'] and f in r['predicted'] for r in rows)
        fn=sum(f in r['expected'] and f not in r['predicted'] for r in rows)
        detail[f]={'true_positive':tp,'false_positive':fp,'false_negative':fn,
                   'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'families':detail,'macro_f1':sum(r['f1'] for r in detail.values())/len(families),
            'exact_set_accuracy':sum(set(r['expected'])==set(r['predicted']) for r in rows)/len(rows)}
def paired_resource_verdict(pairs,parity):
    if not pairs: raise ValueError('No complete paired trials')
    ratios=[]
    for p in pairs:
        vals=[p[k] for k in ('framework_seconds','baseline_seconds','framework_peak_rss','baseline_peak_rss')]
        if any(not isinstance(v,(int,float)) or v<=0 for v in vals): raise ValueError('Missing/nonpositive cost')
        ratios.append({'runtime_ratio':p['framework_seconds']/p['baseline_seconds'],
                       'peak_rss_ratio':p['framework_peak_rss']/p['baseline_peak_rss']})
    joint=parity and all(r['runtime_ratio']<1 and r['peak_rss_ratio']<=1 for r in ratios)
    return {'decision':'OBSERVED_CATALOG_SAFETY_PARITY_WITH_LOWER_RUNTIME_AND_MEMORY' if joint else 'NO_JOINT_RESOURCE_ADVANTAGE',
            'paired_ratios':ratios,'boundary':'Descriptive engineering resource comparison, not independent-study inference or universal accuracy superiority.'}

