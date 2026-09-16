#!/usr/bin/env python3
"""Restore exact archived source and execute RealDefect cases in a fresh workdir.

This wrapper changes no case code, oracle, threshold, or selected input.
RD8's complete scientific negative result returns one; wrapper success records
that expected outcome explicitly rather than interpreting it as reproduction.
"""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,tarfile,time,urllib.request,urllib.parse
import certifi
os.environ.setdefault("SSL_CERT_FILE",certifi.where())

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def log_run(name,argv,cwd,logs,env=None):
 started=datetime.datetime.now(datetime.timezone.utc);base=logs/(started.strftime('%Y%m%dT%H%M%S.%fZ')+'_'+name);start=time.perf_counter()
 with Path(str(base)+'.log').open('w') as f:
  proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
  rc=proc.wait()
 record={'name':name,'argv':argv,'cwd':str(cwd),'started_utc':started.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duration_seconds':time.perf_counter()-start,'exit_code':rc,'output_file':base.name+'.log'}
 dump(Path(str(base)+'.json'),record)
 with (logs/'commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 return rc

def restore(project,work):
 code=work/'code';shutil.copytree(project/'original',code)
 # Keep only input projections/checkpoints; no historical analysis result/state is reused.
 historical=work/'historical';historical.mkdir()
 if (code/'results').exists():shutil.move(str(code/'results'),str(historical/'source-reference-results'))
 (code/'results').mkdir();(code/'runs').mkdir(exist_ok=True)
 shutil.copytree(project/'inputs/rd8',code/'runs/rd8_fresh')
 recovered=[{'path':str(p.relative_to(code)),'sha256':sha(p)} for p in sorted((code/'runs/rd8_fresh').rglob('*')) if p.is_file()]
 assert len(list((code/'runs/rd8_fresh/projections').glob('*.npz')))==123
 # Small declared upstream metadata/fixture inputs are bundled and separately licensed.
 small=project/'reproducibility/light_inputs/data'
 if small.exists():shutil.copytree(small,code/'data',dirs_exist_ok=True)
 dump(work/'restoration.json',{'public_manifest_sha256':sha(project/'MANIFEST.json'),'source_mapping_sha256':sha(project/'SOURCE_MAP.json'),'rd8_input_files':recovered,'historical_analysis_state_reused':False});return code

def inputs(package,code,work,download,reuse,selected):
 specs=[x for x in json.loads((package/'light_inputs/downloads.json').read_text()) if x['case'].split('_')[0] in selected];manifest=json.loads((code/'contracts/real-defect-corpus-manifest.json').read_text());ds=manifest['rd7_groot_172']['dataset']
 for v in ds['selected'] if 'RD7' in selected else []:
  specs.append({'case':'RD7','relative_path':'runs/rd7_downloaded/'+v['path'],'url':'https://huggingface.co/datasets/'+ds['repo_id']+'/resolve/'+ds['revision']+'/'+urllib.parse.quote(v['path'],safe='/'),'expected_sha256':v['lfs_sha256']})
 records=[]
 for spec in specs:
  p=code/spec['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter();method='existing'
  if not p.exists() and reuse is not None and (reuse/spec['relative_path']).exists():shutil.copyfile(reuse/spec['relative_path'],p);method='read-only local input reuse'
  if not p.exists():
   if not download:raise RuntimeError(f'Missing {p}; pass --download or --reuse-input-root pointing at an existing replay code directory')
   with urllib.request.urlopen(spec['url'],timeout=180) as response,p.open('wb') as dest:shutil.copyfileobj(response,dest)
   method='public immutable URL download'
  actual=sha(p);record={**spec,'actual_sha256':actual,'match':actual==spec['expected_sha256'],'bytes':p.stat().st_size,'method':method,'duration_seconds':time.perf_counter()-start};records.append(record);dump(work/'input_verification.json',records)
  if not record['match']:raise RuntimeError(f'Input SHA mismatch: {p}')
 return records

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--download',action='store_true');p.add_argument('--reuse-input-root',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--cases',default='RD1,RD2,RD3,RD4,RD5,RD6,RD7,RD8');a=p.parse_args();project=a.project_root.resolve();package=project/'reproducibility';work=a.work_dir.resolve()
 if work.exists() and any(work.iterdir()):raise RuntimeError('Work directory must be empty to avoid mixing historical/new results')
 work.mkdir(parents=True,exist_ok=True);logs=work/'logs';logs.mkdir();start=time.perf_counter();code=restore(project,work)
 if a.prepare_only:
  dump(work/'prepare_only.json',{'restored_source':str(code),'source_and_archived_projection_hashes_verified':True,'case_executions':0});print('Prepared source and 123 RD8 input projections only; no case executed.');return
 names={'RD1':'run_real_defect_2610.py','RD2':'run_real_defect_2057.py','RD3':'run_real_defect_isaaclab_4559.py','RD4':'run_isaaclab_quaternion_migration.py','RD5':'run_real_defect_robosuite_626.py','RD6':'run_real_defect_openpi_557.py','RD7':'run_real_defect_groot_172.py','RD8':'run_real_defect_openpi_570.py'};selected=a.cases.split(',');assert selected and set(selected)<=set(names);outcomes=[]
 records=inputs(package,code,work,a.download,a.reuse_input_root.resolve() if a.reuse_input_root else None,selected)
 for case in selected:
  argv=[sys.executable,'-B',str(code/'scripts'/names[case])]
  if case=='RD7':argv+=['--video-root',str(code/'runs/rd7_downloaded')]
  if case=='RD8':argv+=['--work-dir',str(code/'runs/rd8_fresh')]
  rc=log_run(case,argv,code,logs);expected=1 if case=='RD8' else 0
  if case=='RD8':
   result=json.loads((code/'results/real-defect-openpi-570.json').read_text());assert result['status']=='NOT_REPRODUCED' and result['correctness']['reproduced'] is False,'RD8 outcome differs; do not change its rule'
  outcomes.append({'case':case,'exit_code':rc,'expected_exit_code':expected,'expected_recorded_outcome':rc==expected})
  if rc!=expected:dump(work/'run_summary.json',{'outcomes':outcomes});raise RuntimeError(f'{case} execution differs from recorded outcome')
  if case in {'RD7','RD8'}:
   stem='groot_172' if case=='RD7' else 'openpi_570';run_dir=code/'runs'/('p13-real-defect-groot-172' if case=='RD7' else 'rd8_fresh');rc=log_run(case+'_audit',[sys.executable,'-B',str(code/'scripts'/('audit_real_defect_'+stem+'.py')),'--run-dir',str(run_dir),'--result',str(code/'results'/('real-defect-groot-172.json' if case=='RD7' else 'real-defect-openpi-570.json'))],code,logs)
   if rc:raise RuntimeError(case+' post-run audit failed')
 dump(work/'run_summary.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outcomes':outcomes,'case_results':{p.name:sha(p) for p in sorted((code/'results').glob('*.json'))},'seconds':time.perf_counter()-start,'interpretation':'expected process outcomes recorded; RD8 remains a scientific negative, not reproduced'})
 print(json.dumps({'work_dir':str(work),'outcomes':outcomes},indent=2))
if __name__=='__main__':main()
