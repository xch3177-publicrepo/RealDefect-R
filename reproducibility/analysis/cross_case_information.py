"""Generate scoped cross-case checks from fresh replay outputs and fixed inputs.
No case-level reproduction decisions or thresholds are changed.
"""
from pathlib import Path
import json,hashlib,datetime,sys,argparse
import numpy as np
import h5py
ROOT=Path.cwd();ap=argparse.ArgumentParser();ap.add_argument('--replay-root',type=Path,default=ROOT/'runs/reproduction_paper_a/code');ap.add_argument('--output',type=Path,default=ROOT/'runs/reproduction_paper_a/cross_case_information.json');args=ap.parse_args();C=args.replay_root.resolve();used={}
def provenance_path(p):return str(p.resolve())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def result(name):
 p=C/'results'/name;used[provenance_path(p)]=sha(p);return json.loads(p.read_text())
rows=[]
d=result('real-defect-lerobot-2610.json');r=d['rows'];a=d['summary']['episode_start'];z=d['summary']['episode_end']
struct={b:all(type(q[b+'_query_index'])==int and a<=q[b+'_query_index']<z and type(q[b+'_is_pad'])==bool for q in r) for b in ['buggy','fixed']}
eq={b:sum(q[b+'_query_index']==q['absolute_index'] for q in r) for b in ['buggy','fixed']}
rows.append({'case':'RD1','structure':{'definition':'integer in-bound query index and Boolean padding flag; does not check padding meaning','count_per_branch':len(r),'passes':struct},'raw_equality':{'definition':'zero-offset returned integer index equals original absolute row index','equal_counts':eq,'count':len(r)},'roundtrip':{'status':'NA','reason':'clamped temporal selection has no specified inverse'},'reference':{'rule':'absolute episode identity and padding meaning','old_violations':d['summary']['buggy_contract_violations'],'fixed_violations':d['summary']['fixed_contract_violations']}})
d=result('real-defect-lerobot-2057.json');struct={}
for b in ['buggy','fixed']:
 struct[b]=all(all(type(r[k])==int and r[k]>=0 for k in ['episode_index','length','dataset_from_index','dataset_to_index']) and r['dataset_to_index']-r['dataset_from_index']==r['length'] for r in d[b+'_rows'])
rows.append({'case':'RD2','structure':{'definition':'nonnegative integer fields and individual end-start=length','count_per_branch':d['summary']['episodes_evaluated'],'passes':struct},'raw_equality':{'status':'NA','reason':'source episode lengths and derived global intervals have different representations'},'roundtrip':{'status':'NA','reason':'no inverse operation specified for metadata accumulation'},'reference':{'rule':'global episode interval continuity','old_violations':len(d['buggy_issues']),'fixed_violations':len(d['fixed_issues'])}})
d=result('real-defect-isaaclab-4559.json');qs={b:np.asarray(d[b+'_xyzw']) for b in ['buggy','fixed']};src=np.asarray(d['legacy_wxyz'])
rows.append({'case':'RD3','structure':{'definition':'width four, finite values, unit norm within 1e-6','count_per_branch':1,'passes':{b:q.shape==(4,) and bool(np.isfinite(q).all()) and abs(float(np.linalg.norm(q))-1)<=1e-6 for b,q in qs.items()}},'raw_equality':{'definition':'literal source WXYZ component array equals output XYZW array without convention decoding','equal':{b:bool(np.array_equal(src,q)) for b,q in qs.items()}},'roundtrip':{'status':'NOT_EVALUATED','reason':'no inverse-conversion experiment executed'},'reference':{'rule':'intended physical orientation','old_error_deg':d['buggy_check']['angular_error_deg'],'fixed_error_deg':d['fixed_check']['angular_error_deg']}})
d=result('real-defect-isaaclab-quaternion-migration.json');sp=C/'data/raw/isaaclab/franka_stack_v51.hdf5';tp=C/'runs/isaaclab-quaternion-migration/native-output.hdf5';used[provenance_path(sp)]=sha(sp);used[provenance_path(tp)]=sha(tp)
pairs=[]
with h5py.File(sp,'r') as s,h5py.File(tp,'r') as t:
 def visit(path,ds):
  if not isinstance(ds,h5py.Dataset):return
  x=np.asarray(ds);y=np.asarray(t[path]);leaf=path.rsplit('/',1)[-1]
  if leaf=='root_pose' and x.shape[-1:]==(7,):x=x[...,3:7];y=y[...,3:7]
  elif path.endswith('/obs/eef_quat') and x.shape[-1:]==(4,):pass
  elif path.endswith('/obs/cube_orientations') and x.shape[-1]%4==0:pass
  elif path.endswith('/obs/object') and x.shape[-1]>=21:x=np.concatenate([x[...,i:i+4] for i in [3,10,17]],axis=0);y=np.concatenate([y[...,i:i+4] for i in [3,10,17]],axis=0)
  else:return
  pairs.append((path,x.reshape(-1,4),y.reshape(-1,4)))
 s.visititems(visit)
count=sum(len(x) for _,x,y in pairs);eq=sum(int(np.all(x==y,axis=1).sum()) for _,x,y in pairs)
rows.append({'case':'RD4','structure':{'definition':'declared quaternion arrays retain shape and finite values; fresh replay format/version baseline passes','count':count,'passes':all(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() for _,x,y in pairs),'format_baseline_pass':d['summary']['schema_baseline_accepted']},'raw_equality':{'definition':'literal source/target quaternion coefficients before convention decoding','equal_instances':eq,'instances':count},'roundtrip':{'status':'NOT_EVALUATED','reason':'no inverse migration run'},'reference':{'rule':'all declared fields receive the required convention conversion','unconverted_instances':d['summary']['missed_quaternion_instances'],'converted_instances':d['summary']['converted_quaternion_instances']}})
d=result('real-defect-robosuite-626.json');rows.append({'case':'RD5','structure':{'definition':'fresh replay input action shape/range baseline only; output structural suite not evaluated','count':d['summary']['public_transitions_replayed'],'passes':d['summary']['input_shape_range_baseline_accepted']},'raw_equality':{'status':'NA','reason':'dimensionless controller actions and world displacements are not raw-comparable'},'roundtrip':{'status':'NOT_EVALUATED','reason':'no inverse-frame roundtrip executed'},'reference':{'rule':'displacement excludes base translation','old_violations':d['summary']['buggy_contract_violations'],'fixed_violations':d['summary']['fixed_contract_violations']}})
d=result('real-defect-openpi-557.json');rows.append({'case':'RD6','structure':{'definition':'two finite normalized anchor outputs, without imposing semantic endpoint bounds','count_per_branch':2,'passes':{b:len(d[b+'_anchor_check']['observed_outputs'])==2 and bool(np.isfinite(d[b+'_anchor_check']['observed_outputs']).all()) for b in ['old','fixed']}},'raw_equality':{'status':'NA','reason':'encoder counts/radians and normalized endpoints use different representations'},'roundtrip':{'status':'PASS_OLD','max_error_rad':max(d['old_self_roundtrip_errors_rad']),'fixed':'NA: whole post-fix policy uses intentionally asymmetric transforms'},'reference':{'rule':'upstream encoder anchors map to normalized zero/one','old_max_error':d['old_anchor_check']['max_abs_error'],'fixed_max_error':d['fixed_anchor_check']['max_abs_error'],'absolute_tolerance':1e-3}})
d=result('real-defect-groot-172.json');rows.append({'case':'RD7','structure':{'definition':'fresh branch return count/shape/dtype/readability checks','videos':d['correctness']['completed_input_count'],'both_branches_pass':d['correctness']['both_branches_structurally_readable_on_all_videos']},'raw_equality':{'status':'NA','reason':'frame correspondence requires a requested-time rule; arbitrary same-position equality is not a baseline'},'roundtrip':{'status':'NA','reason':'timestamp-based selection is many-to-one and has no specified inverse'},'reference':{'rule':'latest source PTS at/before request with declared fallback/ties','old_wrong_noncontrol':sum(d['aggregate']['old']['by_group'][g]['wrong_frame_count'] for g in ['regular','sparse','late']),'fixed_wrong':d['aggregate']['fixed']['all_queries']['wrong_frame_count'],'paired_queries':d['aggregate']['fixed']['all_queries']['query_count']}})
d=result('real-defect-openpi-570.json');struct={b:True for b in ['old','fixed']};failed=0
for c in d['conditions']:
 for b in struct:
  for f in ['state','actions']:
   s=c['branches'][b][f]['statistics'];struct[b]&=len(s['mean'])==len(s['std'])==14 and bool(np.isfinite(s['mean']).all()) and bool(np.isfinite(s['std']).all()) and bool((np.asarray(s['std'])>=0).all())
 failed+=not all(c['branches']['fixed'][f]['oracle_error']['mean_std_within_frozen_tolerance'] for f in ['state','actions'])
rows.append({'case':'RD8','structure':{'definition':'14 finite means/stds and nonnegative std, ignoring population coverage','conditions_per_branch':len(d['conditions']),'passes':struct},'raw_equality':{'status':'NA','reason':'a population and its aggregate statistics have no elementwise raw correspondence'},'roundtrip':{'status':'NA','reason':'statistical aggregation is noninvertible'},'reference':{'rule':'frozen population coverage plus float64 mean/population-std within 1e-5','fixed_full_coverage':d['correctness']['fixed_full_coverage_for_every_order_and_batch'],'fixed_numerical_conditions_failed':failed,'conditions':len(d['conditions'])}})
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'case-local information-requirement comparison; not one uniform generic validator','script_sha256':sha(Path(__file__)),'input_hashes':used,'NA':'no defined appropriate raw identity or inverse comparison for this case operation','NOT_EVALUATED':'a potential check exists but was not executed','rows':rows}
p=args.output.resolve();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cases':len(rows),'RD4_raw_equal':eq,'RD4_total':count,'RD8_fixed_numeric_failures':failed,'output':str(p)},indent=2))
