#!/usr/bin/env python3
"""Verify release hashes and stored result arithmetic without training or network access."""
from pathlib import Path
from collections import Counter
from statistics import mean, stdev
from math import sqrt, isclose
from hashlib import sha256
import ast
import json
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def load(path):return json.loads((ROOT/path).read_text())
def equal(a,b):assert isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def verify():
    exported=load('provenance/export_manifest.json')['files']
    for row in exported:
        path=ROOT/row['path']
        assert sha256(path.read_bytes()).hexdigest()==row['release_sha256'],row['path']
    e3=load('results/cluster/1806501_e3_final/summary.json')
    assert e3['claim_eligible'] is False and e3['test_accessed'] is False
    assert e3['runs']==25 and e3['endpoint_queries']==512
    for key,values in e3['scores_by_condition'].items():
        assert len(values)==5
        equal(mean(values),e3['means_by_condition'][key])
        for i in range(5):
            equal(mean(v[i] for v in e3['task_lag_scores_by_condition'][key].values()),values[i])
        equal(e3['mean_learning_curves'][key][-1]['score'],mean(values))
    for baseline,field in [('random_matched','oracle_vs_random_matched'),('fixed_chunks','oracle_vs_fixed')]:
        deltas=[a-b for a,b in zip(e3['scores_by_condition']['oracle_events'],e3['scores_by_condition'][baseline])]
        summary=e3[field];equal(mean(deltas),summary['mean_delta'])
        half=2.7764451051977987*stdev(deltas)/sqrt(5)
        equal(mean(deltas)-half,summary['paired_t_ci_low']);equal(mean(deltas)+half,summary['paired_t_ci_high'])
        assert sum(v>0 for v in deltas)==summary['positive_seed_count']==4
    expected=[.649609375,.6390625,.901171875,.888671875,.90234375]
    names=['context_only','frame_memory','fixed_chunks','random_matched','oracle_events']
    results={key:[] for key in names}; cohort=None; runs=set()
    for path in sorted((ROOT/'results/cluster/1779882_e2_final').glob('*/seed-*/predictions/*.json')):
        obj=json.loads(path.read_text());rows=obj['predictions']
        assert obj['step']==1000 and len(rows)==512 and obj['claim_eligible'] is False
        # Pointer targets index the condition-specific retrieved records, not a common answer space.
        identity=[(r['query_id'],r['target'] if r['prediction_type']=='answer_vocabulary_index' else None,
                   r['task'],r['effective_lag']) for r in rows]
        if cohort is None:cohort=identity
        assert identity==cohort and len(set(r['query_id'] for r in rows))==512
        assert set(Counter((r['task'],r['effective_lag']) for r in rows).values())=={64}
        assert all(r['correct']==(r['target'] != -100 and r['prediction']==r['target']) for r in rows)
        result=json.loads((path.parent.parent/'result.json').read_text())
        score=mean(r['correct'] for r in rows)
        equal(score,result['validation']['long_delay_memory_score'])
        results[obj['condition']].append(score);runs.add((obj['condition'],obj['seed']))
    assert len(runs)==25
    for name,expected_score in zip(names,expected):assert len(results[name])==5;equal(mean(results[name]),expected_score)
    sherlock=load('results/cluster/1764075/result.json')
    assert sherlock['sealed_test_accessed'] is False and sherlock['claim_eligible'] is False
    contrast=sherlock['primary_development_contrast']
    equal(contrast['human_events']-contrast['random_matched_mean'],contrast['human_minus_random_matched'])
    assert contrast['human_minus_random_matched']<0
    for path in (ROOT/'src').rglob('*.py'):ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)))
    for path in (ROOT/'docs/figures').glob('*.svg'):ET.parse(path)
    # Root and current documentation links must resolve. Historical notes deliberately identify omitted sources.
    for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
        for url in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if re.match(r'https?://|mailto:|#',url):continue
            target=(path.parent/url.split('#')[0]).resolve()
            assert target.exists(),f'Broken local link in {path.name}: {url}'
    full=ROOT/'provenance/release_files.json'
    if full.exists():
        for row in json.loads(full.read_text())['files']:
            assert sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    return {'passed':True,'exported_files_checked':len(exported),'e2_runs_checked':25,
            'e2_final_prediction_rows_checked':12800,'e3_seeds_checked':5,
            'sherlock_primary_null_preserved':True,'training_run':False,
            'limitations':['E3 original prediction files and checkpoint tensors are not included or re-evaluated.',
                           'Full dependency-backed model tests require a separate configured environment.']}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
