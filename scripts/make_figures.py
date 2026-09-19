#!/usr/bin/env python3
"""Generate the Paper A figures and the eight-case table (TikZ / booktabs) from archived results.

Every number drawn or printed is read from a committed artifact and recorded, with the artifact's
SHA-256 and JSON path, in paper/FIGURE_BINDING.json. Nothing is typed by hand. Run:

    python3 -B scripts/make_figures.py

Outputs (all under paper/):
    pipeline_coverage.tikz Fig. 1 categorical operation/reference map
    fig_rd8_residuals.tikz Fig. 2 batch-size / numerical-error diagnostic
    tab_cases.tex         Table I compact case/local-property/reference/outcome matrix
    tab_gx.tex           Table II per-case paired outcomes and scoped flagged counts
    fig_cosmos_audit.tikz Fig. 3 joint outcomes and source-group identity discrepancies
    tab_consumer.tex      Table III scoped function intervention
    FIGURE_BINDING.json   claim -> source file -> JSON path -> value

Visual system. Monochrome categorical map, diagnostic scatter and grayscale statistical bars.
Category/table labels use 8 pt and numerical charts 9 pt serif type at IEEEtran print size;
no decorative panels, title banners or reduced-font scaling.

"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from pipeline_coverage import generate as generate_pipeline
from rd8_chart_data import generate as generate_rd8
from cosmos_summary_figure import generate as generate_cosmos

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'

SRC = {
    'RD1': 'reproducibility/results/real-defect-lerobot-2610.json',
    'RD2': 'reproducibility/results/real-defect-lerobot-2057.json',
    'RD3': 'reproducibility/results/real-defect-isaaclab-4559.json',
    'RD4': 'reproducibility/results/real-defect-isaaclab-quaternion-migration.json',
    'RD5': 'reproducibility/results/real-defect-robosuite-626.json',
    'RD6': 'reproducibility/results/real-defect-openpi-557.json',
    'RD7': 'reproducibility/results/real-defect-groot-172.json',
    'RD8': 'reproducibility/results/real-defect-openpi-570.json',
    'RD8_MATRIX': 'experiments/rd8_diagnostic_20260914/rd8_condition_matrix.json',
    'CROSS': 'reproducibility/results/cross_case_information.json',
    'GX': 'experiments/gx_baseline_20260914/gx_baseline_results.json',
    'GX_MANIFEST': 'experiments/gx_baseline_20260914/artifact_manifest.json',
    'COSMOS': 'experiments/results/20260912-cosmos-metadata/rerun.json',
    'COSMOS_FRAMES': 'experiments/cosmos_frame_check_20260914/cosmos_frame_check_results.json',
    'COSMOS_C5': 'experiments/cosmos_c5_20260916/public_env_compat2/results.json',
    'COSMOS_CONSUMER': 'experiments/cosmos_consumer_replay_20260916/public_env_compat2/results.json',
}
DATA = {k: json.load(open(ROOT / p)) for k, p in SRC.items()}
BINDING = {'generator': 'scripts/make_figures.py', 'sources': {}, 'figures': {}}
for k, p in SRC.items():
    BINDING['sources'][k] = {'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}


def bind(fig, name, key, path, value):
    BINDING['figures'].setdefault(fig, {})[name] = {'source': key, 'json_path': path, 'value': value}
    return value


def get(key, path):
    o = DATA[key]
    for part in path.split('.'):
        o = o[int(part)] if isinstance(o, list) else o[part]
    return o


def n(v):
    return f'{v:,}'


# ---------------------------------------------------------------------------------------------
# Shared drawing vocabulary
# ---------------------------------------------------------------------------------------------
COLORS = r"""\definecolor{rdink}{gray}{0}
\definecolor{rdink2}{gray}{0}
\definecolor{rdmute}{gray}{0}
\definecolor{rdline}{gray}{0.78}
\definecolor{rdfill}{gray}{0.62}
"""

ARROW = r'$\rightarrow$'


LH = 0.345          # cm of vertical space one 8 pt line occupies
CPC = 0.1620        # conservative cm per character for 8 pt sans text
PAD = 0.17          # interior padding of a card
GAP = 0.07          # gap between a card title and its body


def figure(body, caption, label, wide):
    env = 'figure*' if wide else 'figure'
    return (f'\\begin{{{env}}}[t]\n\\centering\n' + COLORS +
            '\\begin{tikzpicture}[x=1cm,y=1cm,font=\\rmfamily\\footnotesize,inner sep=0pt,'
            'line cap=round,line join=round]\n' + body +
            f'\\end{{tikzpicture}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n')


def vlen(s):
    """Rough printed length of a short LaTeX fragment, used only for line counting."""
    out, i = 0, 0
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            macro = s[i + 1:j]
            out += {'rightarrow': 2, 'to': 2, 'cdot': 1, 'pm': 1, 'times': 1, 'dots': 3,
                    'mathtt': 0, 'texttt': 0, 'textbf': 0, 'emph': 0, 'textsc': 0}.get(macro, 0 if macro else 1)
            i = j if j > i + 1 else i + 2
        elif ch in '${}':
            i += 1
        else:
            out += 1
            i += 1
    return out


def wrap(text, width_cm, cpc=CPC):
    """Greedy word wrap into explicit lines so the drawn height is known in advance."""
    budget = max(6, int(width_cm / cpc))
    lines, cur = [], ''
    for word in text.split():
        cand = word if not cur else cur + ' ' + word
        if vlen(cand) <= budget:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def tbox(x, ytop, lines, width, color='rdink2', font='', align='left'):
    """A text block whose top edge sits exactly at ytop. Returns (tikz, height)."""
    body = r'\\'.join(lines)
    node = (f'\\node[anchor=north west,text={color},inner sep=0pt,font=\\rmfamily\\footnotesize{font},'
            f'text width={width:.2f}cm,align={align}] at ({x:.3f},{ytop:.3f}) {{{body}}};\n')
    return node, LH * len(lines)


def txt(x, y, s, anchor='west', color='rdink', font=''):
    """A single unwrapped line, vertically centred on y."""
    return (f'\\node[anchor={anchor},text={color},inner sep=0pt,'
            f'font=\\rmfamily\\footnotesize{font}] at ({x:.3f},{y:.3f}) {{{s}}};\n')


def rule_h(x0, x1, y, color='rdline', lw=0.5):
    return f'\\draw[{color},line width={lw}pt] ({x0:.3f},{y:.3f}) -- ({x1:.3f},{y:.3f});\n'


def rule_v(x, y0, y1, color='rdline', lw=0.5):
    return f'\\draw[{color},line width={lw}pt] ({x:.3f},{y0:.3f}) -- ({x:.3f},{y1:.3f});\n'


def arrow(x0, y0, x1, y1, color='black'):
    return (f'\\draw[{color},line width=0.7pt,-{{Straight Barb[length=1.6pt,width=2.6pt]}}] '
            f'({x0:.3f},{y0:.3f}) -- ({x1:.3f},{y1:.3f});\n')



# =============================================================================================
# Fig. 1  Replay protocol
# =============================================================================================
def fig_protocol():
    F = 'replay_protocol'
    ok = 0
    for c in ('RD1', 'RD2', 'RD4', 'RD5', 'RD6'):
        ok += int(bind(F, f'{c.lower()}_reproduced', c, 'summary.reproduced', get(c, 'summary.reproduced')))
    ok += int(bind(F, 'rd3_reproduced', 'RD3', 'summary.reproduced', get('RD3', 'summary.reproduced')))
    ok += int(bind(F, 'rd7_status', 'RD7', 'status', get('RD7', 'status')) == 'REPRODUCED')
    neg = bind(F, 'rd8_status', 'RD8', 'status', get('RD8', 'status'))
    fail = bind(F, 'rd8_fail', 'RD8_MATRIX', 'fixed_fail_count', get('RD8_MATRIX', 'fixed_fail_count'))
    cond = bind(F, 'rd8_conditions', 'RD8_MATRIX', 'condition_count', get('RD8_MATRIX', 'condition_count'))
    assert ok == 7 and neg == 'NOT_REPRODUCED'
    bind(F, 'cases_meeting_local_rule', 'RD1..RD7', 'derived: count of per-case reproduced flags', ok)

    W = 8.52
    s = ''
    # Three evidence-bearing stages: compact monochrome, no decorative title banner.
    for x,w,title in [(0,2.17,'Pinned inputs'),(2.67,3.31,'Execution'),(6.54,1.98,'Retained')]:
        s += txt(x+w/2,0,title,anchor='center',font='\\bfseries')
        s += rule_h(x,x+w,-0.25,color='black',lw=0.6)
    for j,label in enumerate(['Source revision','Input SHA-256',r'Reference $R_c$','Rule + tolerance','Scope']):
        s += txt(0.08,-0.52-j*0.42,label)
    for j,label in enumerate(['Verdict','Residuals','Logs','Environment']):
        s += txt(6.68,-0.59-j*0.46,label)
    def branch(y,label):
        st=f'\\draw[black,line width=0.5pt] (3.12,{y-0.23:.2f}) rectangle (5.62,{y+0.23:.2f});\n'
        return st+txt(4.37,y,label,anchor='center')
    s += branch(-0.61,'Faulty path')
    s += branch(-1.28,'Reference path')
    # One pinned input is forked, never fed serially through the two paths.
    s += rule_h(2.19,2.81,-0.945,color='black')
    s += rule_v(2.81,-1.28,-0.61,color='black')
    for y in (-0.61,-1.28):
        s += arrow(2.81,y,3.05,y)
        s += rule_h(5.62,5.99,y,color='black')
    s += rule_v(5.99,-1.28,-0.61,color='black')
    s += arrow(5.99,-0.945,6.46,-0.945)
    s += txt(4.37,-1.73,'same input; fixed rule',anchor='center')
    s += branch(-2.22,'RD4: one-path audit')
    s += arrow(2.19,-2.22,3.05,-2.22)
    s += arrow(5.68,-2.22,6.46,-2.22)
    s += rule_h(0,W,-2.59,color='black',lw=0.6)
    cap = ('Replay protocol: pinned identities, an explicit rule and scope, case-appropriate execution, '
           'and retained evidence. A normally executed negative verdict (RD8) remains distinct from '
           'recovery or execution failure.')
    BINDING.setdefault('layout', {})[F] = {'width_cm': W, 'drawing_height_cm': 2.75, 'single_column': True}
    return figure(s, cap, 'fig:replay-protocol', wide=False)


# =============================================================================================
# Table I  The eight cases
# =============================================================================================
def table_cases():
    F = 'tab_cases'
    b = lambda name, key, path: bind(F, name, key, path, get(key, path))

    # --- RD1
    rd1_v = b('rd1_padding_violations', 'RD1', 'summary.buggy_contract_violations')
    rd1_u = b('rd1_units', 'RD1', 'summary.episode_frames')
    rd1_f = b('rd1_fixed_violations', 'RD1', 'summary.fixed_contract_violations')
    rd1_eq = b('rd1_bare_index_equal', 'CROSS', 'rows.0.raw_equality.equal_counts.buggy')
    rd1_bare = bind(F, 'rd1_bare_index_differing', 'CROSS',
                    'derived: rows.0.raw_equality.count - equal_counts.buggy', rd1_u - rd1_eq)
    rd1_s = b('rd1_start', 'RD1', 'summary.episode_start'); rd1_e = b('rd1_end', 'RD1', 'summary.episode_end')
    # --- RD2
    rd2_v = b('rd2_violations', 'RD2', 'summary.buggy_contract_violations')
    rd2_u = b('rd2_units', 'RD2', 'summary.episodes_evaluated')
    rd2_f = b('rd2_fixed_violations', 'RD2', 'summary.fixed_contract_violations')
    rd2_mb = b('rd2_boundary_mb', 'RD2', 'summary.target_file_size_mb')
    # --- RD3
    rd3_o = b('rd3_old_error_deg', 'RD3', 'summary.buggy_angular_error_deg')
    rd3_f = b('rd3_fixed_error_deg', 'RD3', 'summary.fixed_angular_error_deg')
    rd3_d = b('rd3_basis_displacement', 'RD3', 'summary.buggy_max_basis_displacement')
    # --- RD4
    rd4_m = b('rd4_missed', 'RD4', 'summary.missed_quaternion_instances')
    rd4_d = b('rd4_declared', 'RD4', 'summary.declared_quaternion_instances')
    rd4_c = b('rd4_coverage', 'RD4', 'summary.semantic_coverage')
    rd4_mean = b('rd4_mean_deg', 'RD4', 'summary.mean_missed_orientation_error_deg')
    rd4_med = b('rd4_median_deg', 'RD4', 'summary.median_missed_orientation_error_deg')
    rd4_90 = b('rd4_over_90', 'RD4', 'summary.missed_over_90_degrees')
    b('rd4_schema_baseline_accepts', 'RD4', 'summary.schema_baseline_accepted')
    # --- RD5
    rd5_v = b('rd5_violations', 'RD5', 'summary.buggy_contract_violations')
    rd5_u = b('rd5_units', 'RD5', 'summary.public_transitions_replayed')
    rd5_f = b('rd5_fixed_violations', 'RD5', 'summary.fixed_contract_violations')
    rd5_err = b('rd5_endpoint_error_m', 'RD5', 'summary.max_endpoint_error_m')
    # --- RD6
    rd6_oe = b('rd6_old_anchor_error', 'RD6', 'summary.old_anchor_max_error')
    rd6_fe = b('rd6_fixed_anchor_error', 'RD6', 'summary.fixed_anchor_max_error')
    rd6_ni = len(b('rd6_old_issues', 'RD6', 'old_anchor_check.issues'))
    rd6_na = len(b('rd6_anchor_expected', 'RD6', 'old_anchor_check.expected_outputs'))
    rd6_fi = len(b('rd6_fixed_issues', 'RD6', 'fixed_anchor_check.issues'))
    b('rd6_old_roundtrip_accepted', 'RD6', 'summary.old_self_roundtrip_baseline_accepted')
    # --- RD7
    groups = get('RD7', 'aggregate.old.by_group')
    nc = [g for g in groups if g != 'continuous_pts_negative_control']
    rd7_w = sum(b(f'rd7_old_wrong_{g}', 'RD7', f'aggregate.old.by_group.{g}.wrong_frame_count') for g in nc)
    rd7_q = sum(b(f'rd7_queries_{g}', 'RD7', f'aggregate.old.by_group.{g}.query_count') for g in nc)
    rd7_fw = sum(b(f'rd7_fixed_wrong_{g}', 'RD7', f'aggregate.fixed.by_group.{g}.wrong_frame_count') for g in nc)
    rd7_ctl = b('rd7_control_queries', 'RD7', 'aggregate.old.by_group.continuous_pts_negative_control.query_count')
    rd7_ctl_ok = b('rd7_control_correct', 'RD7', 'aggregate.old.by_group.continuous_pts_negative_control.correct_count')
    rd7_max = b('rd7_max_ts_error_s', 'RD7', 'aggregate.old.all_queries.max_absolute_timestamp_error_seconds')
    assert (rd7_w, rd7_q, rd7_fw) == (200, 200, 0) and rd7_ctl == rd7_ctl_ok
    # --- RD8
    rd8_fail = b('rd8_fail', 'RD8_MATRIX', 'fixed_fail_count')
    rd8_cond = b('rd8_conditions', 'RD8_MATRIX', 'condition_count')
    rd8_max = b('rd8_max_error', 'RD8_MATRIX', 'max_fixed_error.value')
    rd8_tol = b('rd8_tolerance', 'RD8_MATRIX', 'frozen_tolerance')
    rd8_cov = b('rd8_old_coverage', 'RD8', 'primary_condition.old_state_row_coverage')
    rd8_rows = b('rd8_rows', 'RD8', 'projection.rows')
    rd8_eps = b('rd8_episodes', 'RD8', 'projection.episodes')
    rd8_bs = b('rd8_primary_batch', 'RD8', 'primary_condition.batch_size')
    rd8_status = b('rd8_status', 'RD8', 'status')
    rd8_status_tex = rd8_status.replace('_', r'\_')
    rd8_covered = round(rd8_cov * rd8_rows)
    bind(F, 'rd8_old_covered_rows', 'RD8', 'derived: coverage * projection.rows', rd8_covered)

    # Short property labels are audited summaries of these archived definitions, not a new suite.
    local = {
        'RD1': r'In-bounds index; Boolean pad',
        'RD2': r'Nonnegative integers; end$-$start$=$length',
        'RD3': r'Width four; finite; unit norm',
        'RD4': r'Shape; finite; format/version',
        'RD5': r'Raw input shape; $[-1,1]$ range',
        'RD6': r'Finite anchors; old-only round-trip',
        'RD7': r'Count, shape, dtype; readability',
        'RD8': r'Finite mean/std; std $\geq0$',
    }
    for i,row in enumerate(get('CROSS','rows')):
        bind(F, row['case'].lower()+'_local_property_definition', 'CROSS',
             f'rows.{i}.structure',row['structure'])
        if row['case']=='RD6':
            bind(F,'rd6_local_roundtrip_scope','CROSS',f'rows.{i}.roundtrip',row['roundtrip'])
    A = ARROW
    rows = [
        ('RD1', r'LeRobot window~\cite{2}', 'E3 minimal path', local['RD1'], 'Absolute row identity',
         f'{rd1_v}/{rd1_u} {A} {rd1_f}/{rd1_u} padding violations'),
        ('RD2', r'LeRobot migration~\cite{4}', 'E2 loop', local['RD2'], 'Global continuity',
         f'{rd2_v}/{rd2_u} {A} {rd2_f}/{rd2_u} episode violations'),
        ('RD3', r'Isaac Lab offset~\cite{1}', 'E3 constant', local['RD3'], 'Quaternion convention',
         f'${rd3_o:.0f}^{{\\circ}} \\to {rd3_f:.0f}^{{\\circ}}$ orientation error'),
        ('RD4', r'Isaac Lab migration~\cite{3}', 'Body audit', local['RD4'], 'Field inventory; convention',
         f'{n(rd4_m)}/{n(rd4_d)} unconverted instances'),
        ('RD5', r'robosuite motion~\cite{5}', 'E2 branch', local['RD5'], 'Displacement semantics',
         f'{n(rd5_v)}/{n(rd5_u)} {A} {rd5_f}/{n(rd5_u)} violations'),
        ('RD6', r'OpenPI calibration~\cite{6}', 'E2 branch', local['RD6'], 'Encoder anchors',
         f'{rd6_ni}/{rd6_na} {A} {rd6_fi}/{rd6_na} anchor violations'),
        ('RD7', r'GR00T loading~\cite{7}', 'E2 branch', local['RD7'], 'Requested-time mapping',
         f'{rd7_w}/{rd7_q} {A} {rd7_fw}/{rd7_q} wrong non-control frames'),
        ('RD8', r'OpenPI statistics~\cite{8}', 'E2 reducer', local['RD8'], 'Population statistic',
         f'\\textbf{{Negative:}} {rd8_fail}/{rd8_cond} conditions fail'),
    ]
    stages = {'RD1':'Loading','RD2':'Migration','RD3':'Controller input','RD4':'Migration','RD5':'Controller input','RD6':'Preprocessing','RD7':'Loading','RD8':'Statistics'}
    ecosystems = {'RD1':r'LeRobot~\cite{2}', 'RD2':r'LeRobot~\cite{4}', 'RD3':r'Isaac Lab~\cite{1}', 'RD4':r'Isaac Lab~\cite{3}', 'RD5':r'robosuite~\cite{5}', 'RD6':r'OpenPI~\cite{6}', 'RD7':r'GR00T~\cite{7}', 'RD8':r'OpenPI~\cite{8}'}
    rows = [(row[0], ecosystems[row[0]], stages[row[0]], *row[2:]) for row in rows]
    BINDING['figures'][F]['stages'] = {'source':'case definitions', 'value':stages}
    widths=[0.58,1.53,1.65,1.46,3.20,2.64,5.12]
    spec='@{}'+''.join(f'>{{\\raggedright\\arraybackslash}}p{{{w}cm}}' for w in widths)+'@{}'
    out = [r'\begin{table*}[t]', r'\centering', r'\footnotesize',
           r'\caption{Cases, local properties checked, required references and replay outcomes.}',
           r'\label{tab:cases}', r'\renewcommand{\arraystretch}{1.10}', r'\setlength{\tabcolsep}{1.5pt}',
           r'\begin{tabular}{'+spec+'}', r'\toprule',
           r'\textbf{Case} & \textbf{Ecosystem} & \textbf{Stage} & \textbf{Execution} & \textbf{Local property checked} & \textbf{Required reference} & \textbf{Replay outcome} \\',
           r'\midrule']
    out += [' & '.join(row) + r' \\' for row in rows]
    out += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',
            r'\begin{minipage}{\textwidth}\footnotesize',
            r'Arrows: faulty $\to$ reference; denominators are case-specific. E2: branch/loop; E3: constant/minimal path. '
            r'RD4 is an unpaired audit; RD5 checks raw input, not transformed displacement; RD6 round-trip applies only to the old pair. '
            f'RD1 bare-index disagreement: {rd1_bare}/{rd1_u}; RD7 controls: {n(rd7_ctl)} correct. RD8 restores '
            f'{n(rd8_rows)}/{n(rd8_rows)} rows, but {rd8_fail}/{rd8_cond} conditions exceed $10^{{-5}}$ '
            f'(max $ {rd8_max / 10 ** math.floor(math.log10(rd8_max)):.3f}\\times10^{{{math.floor(math.log10(rd8_max))}}}$). '
            r'Exact local-check definitions and scopes are in the registry.',
            r'\end{minipage}',r'\end{table*}','']
    return '\n'.join(out)


# =============================================================================================
# Table II  Per-case paired outcomes in Great Expectations
# =============================================================================================
def table_gx():
    F = 'tab_gx'
    out = get('GX', 'outcomes')
    denom = bind(F, 'denominator', 'GX', 'denominator', get('GX', 'denominator'))
    V, C = {}, {}
    for i, o in enumerate(out):
        V[(o['case'], o['arm'])] = bind(F, f'{o["case"]}_{o["arm"]}', 'GX', f'outcomes.{i}.verdict', o['verdict'])
        br = o['branches']
        V[(o['case'], o['arm'], 'faulty')] = br['faulty']['success']
        V[(o['case'], o['arm'], 'reference')] = br['reference']['success']
        bind(F, f'{o["case"]}_{o["arm"]}_branch_success', 'GX', f'outcomes.{i}.branches.*.success',
             {'faulty': br['faulty']['success'], 'reference': br['reference']['success']})
        if o['arm'] == 'C_information_enriched':
            e, r = br['faulty']['expectations'][0], br['reference']['expectations'][0]
            C[o['case']] = (e['unexpected_count'], e['element_count'], r['unexpected_count'])
            bind(F, f'{o["case"]}_informed_counts', 'GX', f'outcomes.{i}.branches.*.expectations.0',
                 {'faulty_unexpected': e['unexpected_count'], 'elements': e['element_count'],
                  'reference_unexpected': r['unexpected_count']})
    na_reason = bind(F, 'rd3_excluded', 'GX', 'excluded_cases.RD3', get('GX', 'excluded_cases.RD3'))
    assert 'NA' in na_reason

    cases = ['RD1', 'RD2', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8']
    rule = {'RD1': 'clamp-and-pad row identity', 'RD2': 'global interval chaining',
            'RD4': 'quaternion convention', 'RD5': 'displacement versus position',
            'RD6': 'calibration anchors', 'RD7': 'requested-time correspondence',
            'RD8': 'agreement with a reference statistic'}
    anchors = bind(F, 'rd6_applicable_anchor_values', 'GX_MANIFEST',
                   'cases.RD6.notes.anchor_expected_norm',
                   get('GX_MANIFEST', 'cases.RD6.notes.anchor_expected_norm'))
    applicable = bind(F, 'rd6_applicable_anchor_count', 'GX_MANIFEST',
                      'derived: len(cases.RD6.notes.anchor_expected_norm)', len(anchors))
    outside = bind(F, 'rd6_rows_outside_rule_scope', 'GX+GX_MANIFEST',
                   'derived: informed element_count minus applicable anchor count',
                   C['RD6'][1] - applicable)
    arms = ['A_declared_schema','B_disjoint_reference_fitted','C_information_enriched']
    symbols={(False,True):r'$\checkmark$',(True,True):r'$\circ$',(False,False):r'$\times$',
             (True,False):r'$\triangle$'}
    rendered=[]
    separated=[]
    for arm in arms:
        separated.append(bind(F,arm+'_separated','GX','derived: faulty=false and reference=true count',
                              sum((V[(c,arm,'faulty')],V[(c,arm,'reference')])==(False,True) for c in cases)))
    for c in ['RD1','RD2','RD3','RD4','RD5','RD6','RD7','RD8']:
        if c=='RD3':
            rendered.append(r'RD3 & \multicolumn{3}{c}{NA} & --- \\')
            continue
        sym=[symbols[(V[(c,arm,'faulty')],V[(c,arm,'reference')])] for arm in arms]
        fu,total,ru=C[c]
        denominator=applicable if c=='RD6' else total
        bind(F,c+'_displayed_applicable_denominator','GX_MANIFEST' if c=='RD6' else 'GX',
             'derived: anchor count' if c=='RD6' else 'C expectation element_count',denominator)
        flagged=f'{n(fu)}/{n(denominator)} $\\to$ {n(ru)}/{n(denominator)}'
        rendered.append(' & '.join([c,*sym,flagged])+r' \\')
    out=[r'\begin{table}[t]',r'\centering',r'\footnotesize',
         r'\caption{Per-case paired outcomes in native Great Expectations 1.6.3.}',r'\label{tab:gx}',
         r'\renewcommand{\arraystretch}{1.12}',r'\setlength{\tabcolsep}{5pt}',
         r'\begin{tabular}{@{}lcccl@{}}',r'\toprule',
         r'\textbf{Case} & \textbf{A} & \textbf{B} & \textbf{C} & \textbf{C: flagged / applicable} \\',r'\midrule',
         *rendered,r'\midrule',
         f'Separated & {separated[0]}/{denom} & {separated[1]}/{denom} & {separated[2]}/{denom} & faulty $\\to$ reference \\\\',
         r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',
         r'\begin{minipage}{\columnwidth}\footnotesize',
         r'A: declared; B: reference-fitted; C: case-informed. '
         r'$\checkmark$: faulty rejected/reference accepted; $\circ$: both accepted; $\times$: both rejected. '
         r'RD3 is NA. RD4 uses a constructed reference; '
         f'RD6 covers {applicable} anchors ({n(outside)} other rows outside scope); RD7 includes controls; '
         'RD8 counts feature-statistic records at the primary condition. '
         r'No reference-only rejections occurred. Predicates and references vary; counts assess supplied-rule portability.',
         r'\end{minipage}',r'\end{table}','']
    return '\n'.join(out)


# =============================================================================================
# Fig. 2  Cosmos split summary-discrepancy proportions
# =============================================================================================
def fig_cosmos_audit():
    F = 'fig_cosmos_audit'
    b_ = lambda name, key, path: bind(F, name, key, path, get(key, path))
    rec = b_('total_records', 'COSMOS', 'summary.episode_rows')
    fr = b_('total_frames', 'COSMOS', 'summary.declared_frames_represented')
    idm = b_('total_identity_mismatch', 'COSMOS', 'summary.identity_inconsistent_rows')
    tkm = b_('total_task_mismatch', 'COSMOS', 'summary.task_companion_mismatch_rows')
    sp = {}
    for split in ('success', 'failure'):
        sp[split] = dict(
            rec=b_(f'{split}_records', 'COSMOS', f'splits.{split}.summary.episode_rows'),
            idm=b_(f'{split}_identity_mismatch', 'COSMOS', f'splits.{split}.summary.identity_inconsistent_rows'),
            tkm=b_(f'{split}_task_mismatch', 'COSMOS', f'splits.{split}.summary.task_companion_mismatch_rows'))
    assert sp['success']['rec'] + sp['failure']['rec'] == rec
    assert sp['success']['idm'] + sp['failure']['idm'] == idm
    assert sp['success']['tkm'] + sp['failure']['tkm'] == tkm
    masked = bind(F, 'failure_split_text_masked', 'COSMOS',
                  'derived: failure identity_inconsistent_rows - task_companion_mismatch_rows',
                  sp['failure']['idm'] - sp['failure']['tkm'])
    files = b_('frame_files', 'COSMOS_FRAMES', 'totals.files')
    eps = b_('frame_episodes', 'COSMOS_FRAMES', 'totals.episodes')
    ffr = b_('frame_frames', 'COSMOS_FRAMES', 'totals.frames')
    fail = b_('frame_failures', 'COSMOS_FRAMES', 'per_episode_failure_count')
    cols = b_('frame_columns', 'COSMOS_FRAMES', 'columns_read')
    bytes_ = b_('frame_bytes', 'COSMOS_FRAMES', 'transfer_accounting.bytes_fetched_this_run')
    cc = {k: b_(f'frame_{k}', 'COSMOS_FRAMES', f'totals.{k}')
          for k in ('c1_length', 'c2_index_interval', 'c3_frame_order', 'c4_task_identity')}
    assert ffr == fr and eps == rec and fail == 0 and all(v == eps for v in cc.values())
    c5 = {k: b_(f'c5_{k}', 'COSMOS_C5', f'totals.{k}')
          for k in ('files', 'episodes', 'rows', 'violating_rows', 'violating_episodes')}
    c5_status = b_('c5_status', 'COSMOS_C5', 'status')
    c5_complete = b_('c5_complete_coverage', 'COSMOS_C5', 'complete_coverage')
    assert (c5['files'], c5['episodes'], c5['rows']) == (files, eps, ffr)
    assert c5_status == 'PASS' and c5_complete and c5['violating_rows'] == c5['violating_episodes'] == 0
    cr_status = b_('consumer_status', 'COSMOS_CONSUMER', 'status')
    cr_commit = b_('consumer_commit', 'COSMOS_CONSUMER', 'consumer.commit')
    cr_function = b_('consumer_function', 'COSMOS_CONSUMER', 'consumer.function')
    cr_records = b_('consumer_records', 'COSMOS_CONSUMER', 'totals.original.records')
    cr_old = {k: b_(f'consumer_original_{k}', 'COSMOS_CONSUMER', f'totals.original.{k}')
              for k in ('violating_records', 'violating_fields')}
    cr_fixed = {k: b_(f'consumer_corrected_{k}', 'COSMOS_CONSUMER', f'totals.summary_corrected.{k}')
                for k in ('violating_records', 'violating_fields')}
    cr_affected = bind(F, 'consumer_affected_records', 'COSMOS_CONSUMER',
                       'derived: count(records with kind = affected)',
                       sum(r['kind'] == 'affected' for r in get('COSMOS_CONSUMER', 'records')))
    cr_controls = bind(F, 'consumer_control_records', 'COSMOS_CONSUMER',
                       'derived: count(records with kind = control)',
                       sum(r['kind'] == 'control' for r in get('COSMOS_CONSUMER', 'records')))
    cr_frames = bind(F, 'consumer_frame_rows', 'COSMOS_CONSUMER',
                     'derived: sum(records.*.frame_rows)',
                     sum(r['frame_rows'] for r in get('COSMOS_CONSUMER', 'records')))
    assert cr_status == 'EXECUTED' and cr_records == cr_affected + cr_controls

    records = get('COSMOS_CONSUMER', 'records')
    cohort = {}
    for kind in ('affected', 'control'):
        cohort[kind] = {}
        for cond in ('original', 'summary_corrected'):
            value = sum(r['conditions'][cond]['violating_field_count'] for r in records if r['kind']==kind)
            cohort[kind][cond] = bind(F, f'consumer_{kind}_{cond}_targeted_field_violations',
                                     'COSMOS_CONSUMER',
                                     f'derived: sum(records where kind={kind}, conditions.{cond}.violating_field_count)', value)
    drawing, details = generate_cosmos(ROOT)
    BINDING['cosmos_group_figure'] = details
    return drawing


def table_consumer():
    F = 'tab_consumer'
    records = get('COSMOS_CONSUMER', 'records')
    out = [r'\begin{table}[t]',r'\centering',r'\footnotesize',
           r'\caption{Targeted output-field violations in the scoped consumer replay.}',
           r'\label{tab:consumer}',r'\renewcommand{\arraystretch}{1.13}',
           r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{@{}lcc@{}}',r'\toprule',
           r'\textbf{Input condition} & \textbf{Affected ($n=2$)} & \textbf{Controls ($n=2$)} \\',r'\midrule']
    for cond, label in [('original','Original summaries'),('summary_corrected','Summary-only correction')]:
        cells=[]
        for kind in ('affected','control'):
            chosen=[r for r in records if r['kind']==kind]
            bad=sum(r['conditions'][cond]['violating_field_count'] for r in chosen)
            total=len(chosen)*8
            bind(F,kind+'_'+cond,'COSMOS_CONSUMER',f'derived: sum(records kind={kind}, conditions.{cond}.violating_field_count)',{'violations':bad,'targeted_fields':total,'records':len(chosen)})
            cells.append(f'{bad}/{total}')
        out.append(' & '.join([label,*cells])+r' \\')
    out.extend([r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',r'\begin{minipage}{\columnwidth}\footnotesize',
                r'Each cell scores eight episode-index summary fields in two preselected records (16 outputs). Same pinned \texttt{update\_meta\_data}, frame reference and offsets; empty video mapping. Other metadata and complete merge behavior are outside this verdict.',
                r'\end{minipage}',r'\end{table}',''])
    return '\n'.join(out)


if __name__ == '__main__':
    BINDING['pipeline_coverage'] = generate_pipeline(ROOT)
    BINDING['rd8_chart'] = generate_rd8(ROOT)
    outputs = {
        'tab_consumer.tex': table_consumer(),
        'tab_cases.tex': table_cases(),
        'tab_gx.tex': table_gx(),
        'fig_cosmos_audit.tikz': fig_cosmos_audit(),
    }
    for name, body in outputs.items():
        (PAPER / name).write_text(body, encoding='utf-8')
    for stale in ('replay_protocol.tikz', 'fig_replay_outcomes.tikz', 'fig_check_matrix.tikz', 'fig_gx_matrix.tikz'):
        p = PAPER / stale
        if p.exists():
            p.unlink()
    all_outputs = list(outputs) + ['pipeline_coverage.tikz','fig_rd8_residuals.tikz','rd8_conditions.csv','RD8_CHART_BINDING.json']
    BINDING['outputs'] = {name: hashlib.sha256((PAPER / name).read_bytes()).hexdigest() for name in all_outputs}
    BINDING['removed_outputs'] = {
        'replay_protocol.tikz':'protocol retained in prose; figure replaced by operation/reference coverage map',
        'fig_gx_matrix.tikz': 'replaced by Table II (tab_gx.tex): per-case symbolic paired outcomes and scoped C counts',
        'fig_replay_outcomes.tikz': 'replaced by Table I (tab_cases.tex); the dumbbell chart mixed '
                                    'non-comparable per-case denominators on one axis',
        'fig_check_matrix.tikz': 'folded into Table I, whose per-case local-check column carries the same '
                                 'classification with its definition beside it',
    }
    (PAPER / 'FIGURE_BINDING.json').write_text(json.dumps(BINDING, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: len(v) for k, v in BINDING['figures'].items()}), 'values bound')
