"""Draw frozen joint outcomes and independently checked source-prefix index offsets using TikZ."""
import csv
import hashlib
import json
from pathlib import Path

ANALYSIS = 'research/20260919-round6-evidence-charts/cosmos_group_analysis/run1'


def generate(root: Path):
    result_path = root/ANALYSIS/'results.json'
    csv_path = root/ANALYSIS/'by_source_prefix.csv'
    split_csv_path = root/ANALYSIS/'by_split_source_prefix.csv'
    data = json.loads(result_path.read_text())
    split_groups = list(csv.DictReader(split_csv_path.open()))
    assert len(split_groups) == 26
    by_split = {split: {r['source_prefix']: r for r in split_groups if r['split'] == split}
                for split in ['success', 'failure']}
    prefixes = sorted(by_split['success'])
    assert prefixes == sorted(by_split['failure'])
    for split, rows in by_split.items():
        assert sum(int(r['episode_rows']) for r in rows.values()) == data['splits'][split]['episode_rows']
        for r in rows.values():
            assert int(r['episode_offset_min']) == int(r['episode_offset_max'])
            assert (int(r['episode_offset_min']) == 0) == (r['source_prefix'] == 'AUTOLab')
    groups = list(csv.DictReader(csv_path.open()))
    assert len(groups) == data['combined']['source_prefix_count'] == 13
    assert sum(int(r['episode_rows']) for r in groups) == data['combined']['episode_rows']
    def node(x,y,text,opts=''):
        return f'\\node[{opts}] at ({x:.4f},{y:.4f}) {{{text}}};\n'
    def line(x0,y0,x1,y1,opts=''):
        return f'\\draw[line width=.5pt,{opts}] ({x0:.4f},{y0:.4f})--({x1:.4f},{y1:.4f});\n'
    def rect(x0,y0,x1,y1,opts):
        return f'\\path[draw=black,line width=.4pt,{opts}] ({x0:.4f},{y0:.4f}) rectangle ({x1:.4f},{y1:.4f});\n'
    s = r'\begin{figure*}[t]'+'\n'+r'\centering'+'\n'
    s += r'\begin{tikzpicture}[x=1cm,y=1cm,font=\rmfamily\fontsize{9}{10.5}\selectfont,inner sep=0pt,text=black]'+'\n'
    s += r'\definecolor{rdIdentity}{HTML}{0072B2}'+'\n'
    s += r'\definecolor{rdJoint}{HTML}{D55E00}'+'\n'
    s += r'\path[use as bounding box] (0,-.33) rectangle (17.35,5.50);'+'\n'
    # Panel (a): mutually exclusive outcomes of the two summary relations.
    x0,x1 = 2.12,7.98
    for v in [0,25,50,75,100]:
        x=x0+(x1-x0)*v/100
        s += line(x,.76,x,4.60,'black!20')
        s += node(x,.55,str(v),'anchor=north')
    s += line(x0,.76,x1,.76)
    s += node((x0+x1)/2,.03,'Episode records (\\%)')
    counts=[]
    for split,y in [('success',3.80),('failure',2.20)]:
        r=data['splits'][split]; total=r['episode_rows']; c=r['counts']
        assert c['task_only']==0 and sum(c[k] for k in ['both_bad','identity_only','neither_bad'])==total
        s+=node(x0-.16,y,split.capitalize()+r' split\\'+f'$n={total:,}$','anchor=east,align=right')
        left=x0
        # Solid / hatched / open remain distinct in grayscale. Color provides
        # a redundant category cue; no text is reversed out of a colored bar.
        for key,opts in [('both_bad','fill=rdJoint!45'),('identity_only','fill=white,pattern=north east lines,pattern color=rdIdentity'),('neither_bad','fill=white')]:
            n=c[key]
            if n == 0:
                continue
            width=(x1-x0)*n/total
            s+=rect(left,y-.20,left+width,y+.20,opts)
            mid=left+width/2
            if width < .4:
                s+=node(mid+.58,y+.65,f'{n:,}')
                s+=line(mid,y+.23,mid+.28,y+.49)
            else:
                s+=node(mid,y+.42,f'{n:,}')
            left+=width
        counts.append({'split':split,'records':total,'counts':c})
    # Short legend below panel; full relation names are in the caption.
    for x,opts,label in [(1.37,'fill=rdJoint!45','Both'),(3.30,'fill=white,pattern=north east lines,pattern color=rdIdentity','Identity only'),(6.20,'fill=white','Neither')]:
        s+=rect(x,1.06,x+.23,1.28,opts)
        s+=node(x+.35,1.17,label,'anchor=west')
    s+=node(.02,5.25,'(a)','anchor=west')
    # Panel (b): position encodes the constant global-minus-local episode offset.
    # Keep the released source-prefix order, not a rank of risk or failure rate.
    gx0,gx1 = 10.55,17.10
    xmax = 60000
    for v in [0,20000,40000,60000]:
        x=gx0+(gx1-gx0)*v/xmax
        s+=line(x,.76,x,5.04,'black!20')
        s+=node(x,.55,str(v//1000),'anchor=north')
    s+=line(gx0,.76,gx1,.76)
    s+=node((gx0+gx1)/2,.03,r'Global $-$ source-local episode index ($10^3$)')
    s+=node(8.85,5.25,'(b)','anchor=west')
    # Marker shape duplicates color; vertically separated points keep the two
    # zero-offset observations visible without perturbing their x coordinates.
    def marker(x,y,split):
        if split == 'success':
            return f'\\draw[draw=rdIdentity,fill=rdIdentity,line width=.5pt] ({x:.4f},{y:.4f}) circle[radius=.048cm];\n'
        return f'\\path[draw=rdJoint,fill=white,line width=.6pt] ({x:.4f},{y+.060:.4f})--({x-.052:.4f},{y-.038:.4f})--({x+.052:.4f},{y-.038:.4f})--cycle;\n'
    for x,split in [(10.75,'success'),(14.10,'failure')]:
        s+=marker(x,5.25,split)
        s+=node(x+.17,5.25,split.capitalize()+' split','anchor=west')
    for i,prefix in enumerate(prefixes):
        y=4.86-i*.325
        s+=node(gx0-.17,y,prefix,'anchor=east')
        for split,dy in [('success',.058),('failure',-.058)]:
            offset=int(by_split[split][prefix]['episode_offset_min'])
            x=gx0+(gx1-gx0)*offset/xmax
            s+=marker(x,y+dy,split)
    # One direct annotation explains why the zero-offset group is exceptional.
    s+=node(gx0+.25,4.86,r'$\Delta=0$: local = global','anchor=west')
    s+=r'\end{tikzpicture}'+'\n'
    s+=(r'\caption{Summary-layer audit of the pinned Cosmos3-DROID release. '
        r'(a) Joint outcomes of identity-summary and task-text relations: both discrepant, identity only, '
        r'or neither. Task-only discrepancies are zero. Success/failure are dataset outcome splits, '
        r'not audit verdicts; labels give exact record counts. '
        r'(b) Global-minus-source-local episode-index offset, $\Delta$, by source prefix and split: '
        r'$\mathrm{global}=\mathrm{source\mbox{-}local}+\Delta$. Each point is the constant offset '
        r'for all records in that prefix/split; slight vertical separation makes coincident points visible. '
        r'Only AUTOLab has $\Delta=0$ in both splits (10,405 records); all 61,502 records in the other '
        r'12 prefixes have positive offsets and identity-summary disagreement. Offset magnitude reflects '
        r'numbering position, not defect severity. Prefixes are not an independently verified inventory '
        r'of laboratories. These checks do not evaluate visual content or downstream harm.}'+'\n'
        r'\label{fig:cosmos-audit}'+'\n'+r'\end{figure*}'+'\n')
    return s, {'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [result_path,csv_path,split_csv_path]},
               'joint_counts':counts,'source_prefix_rows':groups,
               'episode_offsets':split_groups,'offset_axis_max':xmax,'prefix_order':prefixes,'drawing_width_cm':17.35,
               'drawing_height_cm':5.83,'font_size_pt':9}
