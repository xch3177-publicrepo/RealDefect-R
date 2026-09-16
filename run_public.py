#!/usr/bin/env python3
"""Run a fresh copy of the public package; immutable shipped results remain untouched.

Exit0 means expected scientific outcomes; RD8 NOT_REPRODUCED remains negative evidence.
Infrastructure/installation/download failures exit2, with a separate explicit failure record.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import certifi
class ScientificMismatch(RuntimeError): pass
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--suite',choices=['smoke','gx','cases','extended','all'],default='smoke');p.add_argument('--download',action='store_true');a=p.parse_args()
 out=a.work_dir.resolve()
 if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new empty work directory')
 if out==ROOT or ROOT in out.parents:raise RuntimeError('Work directory must be outside release snapshot')
 out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir();runtime=out/'runtime';records=[];checks=[]
 env=dict(os.environ);env['SSL_CERT_FILE']=certifi.where();env['HF_HUB_DISABLE_TELEMETRY']='1';env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';env.pop('HF_TOKEN',None);env.pop('HUGGING_FACE_HUB_TOKEN',None);env['PYTHONDONTWRITEBYTECODE']='1'
 env['HF_HOME']=str(out/'hf-cache');env['XDG_CACHE_HOME']=str(out/'cache');env['MPLCONFIGDIR']=str(out/'mpl')
 def run(name,args,cwd=runtime):
  start=time.monotonic();path=logs/(name+'.log')
  with path.open('w') as f:
   x=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
  rec={'step':name,'argv':[str(v) for v in args],'exit_code':x.returncode,'seconds':time.monotonic()-start,'log':path.name};records.append(rec);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps(rec),flush=True)
  if x.returncode:raise RuntimeError(f'{name} failed; see {path}')
 def py(name,*args):run(name,[sys.executable,'-B',*args])
 def scientific(name,got,want):
  match=got==want;checks.append({'check':name,'matches':match,'actual':got,'expected':want})
  (out/'SCIENTIFIC_COMPARISON.json').write_text(json.dumps(checks,indent=2)+'\n')
  if not match:raise ScientificMismatch(name+' differs from expected scientific result')
 def load(path):return json.loads(Path(path).read_text())
 def compare_fields(name,fresh,frozen,fields):
  x,y=load(fresh),load(frozen)
  for field in fields:scientific(name+'.'+field,x[field],y[field])
 try:
  run('integrity',[sys.executable,str(ROOT/'verify_public.py')],ROOT)
  shutil.copytree(ROOT,runtime,ignore=shutil.ignore_patterns('.git','.venv','__pycache__'))
  py('projection_recovery','experiments/rd8_diagnostic_20260914/recover_projections.py')
  if a.suite!='gx':
   py('RD3_RD6_scoped_smoke','reproducibility/reproduce.py','--work-dir',str(out/'cases-smoke'),'--cases','RD3,RD6')
   py('RD8_condition_matrix','experiments/rd8_diagnostic_20260914/extract_rd8_condition_matrix.py')
   py('figure_generation','scripts/make_figures.py')
  if a.suite in {'cases','all'}:
   cmd=['reproducibility/reproduce.py','--work-dir',str(out/'cases-all')]
   if a.download:cmd+=['--download']
   py('eight_cases',*cmd)
   rd8=load(out/'cases-all/code/results/real-defect-openpi-570.json')
   failures=sum(not all(c['branches']['fixed'][f]['oracle_error']['mean_std_within_frozen_tolerance'] for f in ['state','actions']) for c in rd8['conditions'])
   scientific('RD8.conditions',len(rd8['conditions']),44);scientific('RD8.numerical_failures',failures,24)
   scientific('RD8.fixed_full_coverage',rd8['correctness']['fixed_full_coverage_for_every_order_and_batch'],True)
   py('cross_case_information','reproducibility/analysis/cross_case_information.py','--replay-root',str(out/'cases-all/code'),'--output',str(out/'cross-case.json'))
   compare_fields('cross_case',out/'cross-case.json',ROOT/'reproducibility/results/cross_case_information.json',['rows'])
  if a.suite in {'gx','extended','all'}:
   if not a.download:raise RuntimeError('GX and extended experiments need public input recovery; use --download')
   specs=json.loads((runtime/'reproducibility/light_inputs/downloads.json').read_text())
   for spec in specs:
    if spec['case'].split('_')[0] not in {'RD4','RD5'}:continue
    target=runtime/'experiments/work/gx_baseline_20260914/inputs'/Path(spec['relative_path']).name;target.parent.mkdir(parents=True,exist_ok=True)
    run('download_'+target.name,['curl','-fLsS','--retry','3',spec['url'],'-o',str(target)])
    if hashlib.sha256(target.read_bytes()).hexdigest()!=spec['expected_sha256']:raise RuntimeError('GX input hash mismatch')
   py('GX_fixture_build','experiments/gx_baseline_20260914/build_artifacts.py')
   py('GX_validation','experiments/gx_baseline_20260914/run_gx_baseline.py','--amend','Public snapshot clean rerun: regenerated fixture manifest; scientific rules unchanged')
   compare_fields('GX',runtime/'experiments/gx_baseline_20260914/gx_baseline_results.json',ROOT/'experiments/gx_baseline_20260914/gx_baseline_results.json',['tally','cases_evaluated','denominator'])
   current_plan=load(runtime/'experiments/gx_baseline_20260914/FROZEN_PLAN.json');original_plan=load(ROOT/'experiments/gx_baseline_20260914/FROZEN_PLAN.json')
   for field in ['arms','cases','suites_sha256']:scientific('GX.preserved_plan.'+field,current_plan[field],original_plan[field])
   if a.suite=='gx':
    # Check the requested Cartesian execution coverage without changing native GX rules.
    result=load(runtime/'experiments/gx_baseline_20260914/gx_baseline_results.json')
    branches=[]
    for outcome in result['outcomes']:
     for branch,details in outcome['branches'].items():
      branches.append({'case':outcome['case'],'arm':outcome['arm'],'branch':branch,'success':details['success'],'expectation_count':details['expectation_count'],'rows_validated':details['rows_validated']})
    wanted={(case,arm,branch) for case in original_plan['cases'] for arm in original_plan['arms'] for branch in ['faulty','reference']}
    observed=[(row['case'],row['arm'],row['branch']) for row in branches]
    coverage={'snapshot_manifest_sha256':hashlib.sha256((ROOT/'MANIFEST.json').read_bytes()).hexdigest(),'expected_branch_count':len(wanted),'executed_branch_count':len(observed),'complete':len(observed)==len(wanted) and set(observed)==wanted and all(type(row['success']) is bool and row['expectation_count']>0 for row in branches),'branches':branches,'scientific_comparison_count':len(checks),'all_scientific_comparisons_match':all(row['matches'] for row in checks),'scope':'native GX only: seven pairs, three arms and both branches; not all eight case replays or the extended suite'}
    (out/'GX_EXECUTION_COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
    if not coverage['complete']:raise ScientificMismatch('GX Cartesian execution coverage incomplete')
  if a.suite in {'extended','all'}:
   py('RD8_precision_diagnostic','experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic.py','all_orders')
   diag=load(runtime/'experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic_all_orders.json');expected=load(ROOT/'experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic_all_orders.json')
   scientific('RD8PD.condition_counts',{k:{q:v[q] for q in ['conditions','within_1e_5','above_1e_5']} for k,v in diag['arm_rollup'].items()},{k:{q:v[q] for q in ['conditions','within_1e_5','above_1e_5']} for k,v in expected['arm_rollup'].items()})
   scientific('RD8PD.D1_exact_matches',diag['d1_vs_frozen_rd8']['exact_matches'],44)
   py('Cosmos_metadata','experiments/replay_cosmos_metadata.py','--out',str(out/'cosmos-metadata'))
   py('Cosmos_independent_metadata','experiments/verify_cosmos_referential_counts.py','--manifest',str(out/'cosmos-metadata/preaudit_verified_inputs.json'),'--out',str(out/'cosmos-independent.json'))
   compare_fields('Cosmos_metadata',out/'cosmos-metadata/rerun.json',ROOT/'experiments/results/20260912-cosmos-metadata/rerun.json',['summary'])
   compare_fields('Cosmos_independent',out/'cosmos-independent.json',ROOT/'experiments/results/20260912-cosmos-metadata/independent_referential_check.json',['splits'])
   py('Cosmos_C1_C4','experiments/cosmos_frame_check_20260914/run_frame_check.py')
   compare_fields('Cosmos_C1_C4',runtime/'experiments/cosmos_frame_check_20260914/cosmos_frame_check_results.json',ROOT/'experiments/cosmos_frame_check_20260914/cosmos_frame_check_results.json',['totals','summary_versus_frames','processed_files','per_episode_failure_count'])
   c5=runtime/'experiments/cosmos_c5_20260916/run_c5.py'
   if c5.exists():
    py('Cosmos_C5',str(c5),'--metadata-cache',str(runtime/'experiments/work/cosmos3-droid'),'--out',str(out/'cosmos-c5'))
    compare_fields('Cosmos_C5',out/'cosmos-c5/results.json',ROOT/'experiments/cosmos_c5_20260916/run_20260916T033332Z/results.json',['totals','status','complete_coverage'])
   consumer=runtime/'experiments/cosmos_consumer_replay_20260916/run_consumer.py'
   if consumer.exists():
    py('Cosmos_consumer',str(consumer),'--metadata-cache',str(runtime/'experiments/work/cosmos3-droid'),'--out',str(out/'cosmos-consumer'))
    compare_fields('Cosmos_consumer',out/'cosmos-consumer/results.json',ROOT/'experiments/cosmos_consumer_replay_20260916/run_20260916T033508Z/results.json',['totals','status'])
  summary={'status':'EXPECTED_SCIENTIFIC_OUTCOMES','suite':a.suite,'executed_steps':[x['step'] for x in records],'RD8':'NOT_REPRODUCED is the expected scientific negative, not infrastructure failure','created_utc':datetime.now(timezone.utc).isoformat()};rc=0
 except Exception as e:
  summary={'status':('SCIENTIFIC_MISMATCH' if isinstance(e,ScientificMismatch) else 'INFRASTRUCTURE_OR_EXECUTION_FAILURE'),'suite':a.suite,'error':str(e),'executed_steps':[x['step'] for x in records],'created_utc':datetime.now(timezone.utc).isoformat()};rc=2
 (out/'RUN_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return rc
if __name__=='__main__':raise SystemExit(main())
