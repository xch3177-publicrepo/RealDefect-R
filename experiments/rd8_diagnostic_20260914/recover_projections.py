#!/usr/bin/env python3
"""Prepare the bundled, separately licensed RD8 projections; no author-local archive required."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 source=ROOT/'inputs/rd8';dest=ROOT/'experiments/work/rd8_diagnostic_20260914'
 manifest=json.loads((ROOT/'MANIFEST.json').read_text())
 expected={x['path']:x['sha256'] for x in manifest['files']}
 for p in sorted(source.rglob('*')):
  if p.is_file():
   assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[str(p.relative_to(ROOT))],p
   target=dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 frozen=json.loads((ROOT/'reproducibility/results/real-defect-openpi-570.json').read_text())['projection']['logical_inventory']
 by_ep={x['episode']:x for x in frozen};count=0
 for p in sorted((dest/'projections').glob('*.npz')):
  ep=int(''.join(c for c in p.stem if c.isdigit()))
  with np.load(p,allow_pickle=False) as arrays:
   for entry in by_ep[ep]['arrays']:
    arr=arrays[entry['name']]
    assert str(arr.dtype)==entry['dtype'] and list(arr.shape)==entry['shape']
    assert hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==entry['logical_sha256']
  count+=1
 assert count==123
 print(json.dumps({'projection_files':count,'canonical_array_hashes_verified':True,'historical_analysis_state_reused':False}))
if __name__=='__main__':main()
