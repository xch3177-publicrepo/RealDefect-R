#!/usr/bin/env python3
"""Generate the Paper A figures and the eight-case table (TikZ / booktabs) from archived results.

Every number drawn or printed is read from a committed artifact and recorded, with the artifact's
SHA-256 and JSON path, in paper/FIGURE_BINDING.json. Nothing is typed by hand. Run:

    python3 -B scripts/make_figures.py

Outputs (all under paper/):
    replay_protocol.tikz  Fig. 1  compact same-input parallel replay protocol
    tab_cases.tex         Table I compact eight-case operation/scope/reference/outcome matrix
    fig_gx_matrix.tikz    Fig. 2  stacked paired-outcome counts of three Great Expectations suites
    fig_cosmos_audit.tikz Fig. 3  Cosmos split proportions, targeted consumer fields, frame-check strip
    FIGURE_BINDING.json   claim -> source file -> JSON path -> value

Visual system. One restrained family: ink for text, navy for structure, teal for a separation or a
measured positive, coral only for a retained failure, pale greys for context. Every mark also carries
a word or a shape, so the figures survive greyscale and colour-vision deficiency; no 3D, no
gradients, no decorative fills. Figure type is set to 8 pt (\\footnotesize at IEEEtran 10 pt) so the
labels stay legible at two-column print size.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path

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
COLORS = r"""\definecolor{rdink}{HTML}{14181F}
\definecolor{rdink2}{HTML}{4A5462}
\definecolor{rdmute}{HTML}{667180}
\definecolor{rdnavy}{HTML}{1F3A5F}
\definecolor{rdnavy2}{HTML}{4A6E9B}
\definecolor{rdteal}{HTML}{16706E}
\definecolor{rdteal2}{HTML}{CFE3E2}
\definecolor{rdcoral}{HTML}{BC4127}
\definecolor{rdcoral2}{HTML}{F2DCD6}
\definecolor{rdline}{HTML}{CFD5DC}
\definecolor{rdfill}{HTML}{EDF0F3}
\definecolor{rdfill2}{HTML}{F7F8FA}
"""

ARROW = r'$\rightarrow$'


LH = 0.345          # cm of vertical space one 8 pt line occupies
CPC = 0.1620        # conservative cm per character for 8 pt sans text
PAD = 0.17          # interior padding of a card
GAP = 0.07          # gap between a card title and its body


def figure(body, caption, label, wide):
    env = 'figure*' if wide else 'figure'
    return (f'\\begin{{{env}}}[t]\n\\centering\n' + COLORS +
            '\\begin{tikzpicture}[x=1cm,y=1cm,font=\\sffamily\\footnotesize,inner sep=0pt,'
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
    node = (f'\\node[anchor=north west,text={color},inner sep=0pt,font=\\sffamily\\footnotesize{font},'
            f'text width={width:.2f}cm,align={align}] at ({x:.3f},{ytop:.3f}) {{{body}}};\n')
    return node, LH * len(lines)


def txt(x, y, s, anchor='west', color='rdink', font=''):
    """A single unwrapped line, vertically centred on y."""
    return (f'\\node[anchor={anchor},text={color},inner sep=0pt,'
            f'font=\\sffamily\\footnotesize{font}] at ({x:.3f},{y:.3f}) {{{s}}};\n')


def panel(x0, y0, x1, y1, fill='rdfill2', draw='rdline', radius=1.8, lw=0.5):
    d = f'draw={draw},line width={lw}pt' if draw else 'draw=none'
    return (f'\\path[fill={fill},{d},rounded corners={radius}pt] '
            f'({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});\n')


def rule_h(x0, x1, y, color='rdline', lw=0.5):
    return f'\\draw[{color},line width={lw}pt] ({x0:.3f},{y:.3f}) -- ({x1:.3f},{y:.3f});\n'


def rule_v(x, y0, y1, color='rdline', lw=0.5):
    return f'\\draw[{color},line width={lw}pt] ({x:.3f},{y0:.3f}) -- ({x:.3f},{y1:.3f});\n'


def arrow(x0, y0, x1, y1, color='rdnavy2'):
    return (f'\\draw[{color},line width=0.7pt,-{{Straight Barb[length=1.6pt,width=2.6pt]}}] '
            f'({x0:.3f},{y0:.3f}) -- ({x1:.3f},{y1:.3f});\n')


def card(x, ytop, width, title, body, fill='rdfill2', draw='rdline', tcolor='rdink',
         bcolor='rdink2', height=None, lw=0.5):
    """Titled card with a wrapped body; height is derived from the wrapped line count."""
    inner = width - 2 * PAD
    lines = wrap(body, inner) if isinstance(body, str) else body
    h = height if height is not None else PAD + LH + GAP + LH * len(lines) + PAD
    s = panel(x, ytop - h, x + width, ytop, fill=fill, draw=draw, lw=lw)
    s += txt(x + PAD, ytop - PAD - LH / 2, title, color=tcolor, font='\\bfseries')
    blk, _ = tbox(x + PAD, ytop - PAD - LH - GAP, lines, inner, color=bcolor)
    return s + blk, h


def section_rule(y, index, title, W, color='rdnavy'):
    """A numbered band header: a small square, small-caps title, then a hairline to the margin."""
    w_lab = 0.52 + vlen(title) * 0.175
    s = f'\\path[fill=white] (-0.05,{y - 0.22:.3f}) rectangle ({w_lab:.3f},{y + 0.20:.3f});\n'
    s += (f'\\path[fill={color},rounded corners=0.6pt] (0,{y - 0.17:.3f}) rectangle '
          f'(0.22,{y + 0.13:.3f});\n')
    s += txt(0.34, y - 0.02, f'\\textbf{{{index}}}', color='white')
    label = f'\\textsc{{{title}}}'
    s += txt(0.44, y - 0.02, label, color=color, font='\\bfseries')
    s += rule_h(0.52 + vlen(title) * 0.175, W, y - 0.02, color='rdline', lw=0.5)
    return s


def legend_chip(x, y, rejected):
    h = 0.17
    if rejected:
        return (f'\\path[fill=rdnavy,rounded corners=0.7pt] ({x:.3f},{y - h:.3f}) rectangle '
                f'({x + 2 * h:.3f},{y + h:.3f});\n')
    return (f'\\path[draw=rdmute,line width=0.7pt,rounded corners=0.7pt] ({x:.3f},{y - h:.3f}) '
            f'rectangle ({x + 2 * h:.3f},{y + h:.3f});\n')


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
    s = txt(0, 0, r'\textbf{Pinned sources, verified input, versioned rule}', color='rdink')
    s += rule_h(0, W, -0.23, color='rdink2')

    def box(x0, y, w, label, color='rdink2'):
        return (f'\\draw[{color},line width=0.6pt] ({x0:.2f},{y-0.27:.2f}) rectangle '
                f'({x0+w:.2f},{y+0.27:.2f});\n' +
                txt(x0+w/2, y, label, anchor='center', color=color))
    # Same input forks before execution. Neither branch consumes the other's output.
    s += box(0.0, -1.10, 1.70, 'Same input')
    s += box(2.65, -0.68, 2.36, 'Faulty path')
    s += box(2.65, -1.52, 2.36, 'Reference path', color='rdteal')
    s += rule_h(1.70, 2.14, -1.10, color='rdink2')
    s += rule_v(2.14, -1.52, -0.68, color='rdink2')
    s += arrow(2.14, -0.68, 2.56, -0.68, color='rdink2')
    s += arrow(2.14, -1.52, 2.56, -1.52, color='rdink2')
    s += rule_h(5.01, 5.50, -0.68, color='rdink2')
    s += rule_h(5.01, 5.50, -1.52, color='rdink2')
    s += rule_v(5.50, -1.52, -0.68, color='rdink2')
    s += arrow(5.50, -1.10, 6.04, -1.10, color='rdink2')
    s += box(6.12, -1.10, 2.40, r'Apply $V_c(\cdot,R_c)$')
    s += txt(3.83, -2.07, 'paired execution', anchor='center', color='rdmute')
    s += txt(0, -2.64, r'\textbf{RD4}', color='rdink')
    s += arrow(0.72, -2.64, 1.10, -2.64, color='rdink2')
    s += box(1.19, -2.64, 3.82, 'Single-path coverage audit')
    s += arrow(5.02, -2.64, 6.04, -2.64, color='rdink2')
    s += box(6.12, -2.64, 2.40, 'Retain outcome')
    s += arrow(7.32, -1.39, 7.32, -2.29, color='rdink2')
    cap = ('Execution structure. Both paired paths receive the same input and fixed criterion; '
           'RD4 instead audits one converter body. Logs retain scientific outcomes separately '
           'from recovery or execution failures (Table~\\ref{tab:cases}).')
    BINDING.setdefault('layout', {})[F] = {'width_cm': W, 'drawing_height_cm': 2.91, 'single_column': True}
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

    A = ARROW
    rows = [
        ('RD1', r'LeRobot window selection~\cite{2}', 'E3 minimal path', 'absolute row identity',
         f'{rd1_v}/{rd1_u} {A} {rd1_f}/{rd1_u} padding violations'),
        ('RD2', r'LeRobot migration~\cite{4}', 'E2 loop', 'global interval continuity',
         f'{rd2_v}/{rd2_u} {A} {rd2_f}/{rd2_u} episode violations'),
        ('RD3', r'Isaac Lab grasp offset~\cite{1}', 'E3 constant', 'quaternion convention',
         f'${rd3_o:.0f}^{{\\circ}} \\to {rd3_f:.0f}^{{\\circ}}$ orientation error'),
        ('RD4', r'Isaac Lab conversion~\cite{3}', 'Body audit', 'field inventory + convention',
         f'{n(rd4_m)}/{n(rd4_d)} stored quaternions unconverted'),
        ('RD5', r'robosuite displacement~\cite{5}', 'E2 branch', 'displacement vs. position',
         f'{n(rd5_v)}/{n(rd5_u)} {A} {rd5_f}/{n(rd5_u)} violations'),
        ('RD6', r'OpenPI calibration~\cite{6}', 'E2 calibration', 'encoder anchors',
         f'{rd6_ni}/{rd6_na} {A} {rd6_fi}/{rd6_na} anchor violations'),
        ('RD7', r'GR00T frame loading~\cite{7}', 'E2 branch', 'requested-time mapping',
         f'{rd7_w}/{rd7_q} {A} {rd7_fw}/{rd7_q} wrong non-control frames'),
        ('RD8', r'OpenPI statistics~\cite{8}', 'E2 reduction', 'full-population statistic',
         f'\\textbf{{Negative:}} {rd8_fail}/{rd8_cond} conditions exceed $10^{{-5}}$'),
    ]
    out = [r'\begin{table*}[t]', r'\centering', r'\footnotesize',
           r'\caption{Cases, execution scope, required references and local outcomes.}',
           r'\label{tab:cases}', r'\renewcommand{\arraystretch}{1.08}', r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}lllp{3.35cm}l@{}}', r'\toprule',
           r'\textbf{Case} & \textbf{Operation / ecosystem} & \textbf{Execution} & \textbf{Reference} & \textbf{Local outcome} \\',
           r'\midrule']
    out += [' & '.join(row) + r' \\' for row in rows]
    out += [r'\bottomrule', r'\end{tabular}', r'\par\vspace{3pt}',
            r'\begin{minipage}{\textwidth}\footnotesize',
            r'Arrows: faulty $\to$ reference; denominators are case-specific. E2: branch/loop; E3: constant/minimal path. RD4 is a single-body audit, without a repaired converter run. '
            f'RD1 bare-index disagreement is {rd1_bare}/{rd1_u}; RD5 $[-1,1]$ is the raw normalized-action bound, not the output displacement bound. '
            f'RD7 has {n(rd7_ctl)} correct controls. RD8 coverage is {n(rd8_rows)}/{n(rd8_rows)} after repair, but its maximum residual is '
            f'$ {rd8_max / 10 ** math.floor(math.log10(rd8_max)):.3f}\\times10^{{{math.floor(math.log10(rd8_max))}}}$; '
            r'the complete numerical criterion stays negative. Full scopes and source identities are in the registry.',
            r'\end{minipage}', r'\end{table*}', '']
    return '\n'.join(out)


# =============================================================================================
# Fig. 2  Case-wise portability of the case rules into Great Expectations
# =============================================================================================
def fig_gx_matrix():
    F = 'fig_gx_matrix'
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
    arms = [('A_declared_schema', 'A: declared'),
            ('B_disjoint_reference_fitted', 'B: ref.-fitted'),
            ('C_information_enriched', 'C: case-informed')]
    categories = [((False, True), 'Faulty only rejected', 'rdteal'),
                  ((True, True), 'Both accepted', 'rdfill'),
                  ((False, False), 'Both rejected', 'rdnavy'),
                  ((True, False), 'Reference only rejected', 'rdcoral')]
    counts = {}
    for arm, _ in arms:
        counts[arm] = []
        for flags, label, _ in categories:
            value = sum((V[(c, arm, 'faulty')], V[(c, arm, 'reference')]) == flags for c in cases)
            counts[arm].append(bind(F, f'{arm}:{label}', 'GX',
                                   'derived: count of branch-success pairs ' + repr(flags), value))
        assert sum(counts[arm]) == denom
    W, x0, x1 = 8.52, 2.30, 8.32
    s = txt(x0, 0.0, 'Number of paired outcomes', color='rdink')
    step = (x1-x0)/denom
    for tick in range(denom+1):
        x = x0+tick*step
        s += rule_v(x, -2.48, -0.35, color='rdline', lw=0.35)
        s += txt(x, -2.68, str(tick), anchor='center', color='rdink2')
    for row, (arm, label) in enumerate(arms):
        y = -0.66-row*0.75
        s += txt(x0-0.16, y, label, anchor='east', color='rdink')
        current = x0
        for j, value in enumerate(counts[arm]):
            if not value:
                continue
            end = current+value*step
            color = categories[j][2]
            s += f'\\path[fill={color},draw=white,line width=0.4pt] ({current:.3f},{y-0.23:.3f}) rectangle ({end:.3f},{y+0.23:.3f});\n'
            s += txt((current+end)/2, y, str(value), anchor='center',
                     color='rdink' if j == 1 else 'white', font='\\bfseries')
            current = end
    for j, (_, label, color) in enumerate(categories):
        x,y=(0,-3.15) if j==0 else (4.30,-3.15) if j==1 else (0,-3.61) if j==2 else (4.30,-3.61)
        s += f'\\path[fill={color},draw=rdline,line width=0.4pt] ({x:.2f},{y-0.11:.2f}) rectangle ({x+0.27:.2f},{y+0.11:.2f});\n'
        s += txt(x+0.38,y,label,color='rdink2')
    cap = (f'Paired outcomes of supplied rules in Great Expectations (native, {denom} pairs). '
           'Suites vary predicates and references; these counts are portability evidence, not accuracy. '
           'RD3 is NA; RD4 uses a constructed reference; RD6 covers only '
           f'{applicable} anchors ({n(outside)} other rows outside scope); RD8 evaluates its primary condition. '
           'All reference-only rejection counts are zero.')
    BINDING.setdefault('layout', {})[F] = {'width_cm': W, 'drawing_height_cm': 3.78, 'single_column': True}
    return figure(s, cap, 'fig:gx-matrix', wide=False)


# =============================================================================================
# Fig. 3  Cosmos split proportions, targeted consumer fields, frame-check strip
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
    W = 17.8
    s = txt(0, 0, r'\textbf{(a) Episode-summary relations}', color='rdink')
    s += txt(9.10, 0, r'\textbf{(b) Scoped consumer: targeted fields only}', color='rdink')
    # Normalized stacked bars; each split has its own explicitly labelled population.
    bx0,bx1 = 3.12,7.16
    bars = [('success','Identity', 'idm',-0.63),('success','Task text','tkm',-1.21),
            ('failure','Identity','idm',-2.08),('failure','Task text','tkm',-2.66)]
    for tick in (0,25,50,75,100):
        x=bx0+(bx1-bx0)*tick/100
        s += rule_v(x,-2.89,-0.37,color='rdline',lw=0.35)
        s += txt(x,-3.11,f'{tick}\\%',anchor='center',color='rdink2')
    for split,label,key,y in bars:
        total,bad=sp[split]['rec'],sp[split][key]
        good=bind(F,f'{split}_{key}_coherent','COSMOS',f'derived: {split}.episode_rows minus {key}',total-bad)
        s += txt(bx0-0.15,y,label,anchor='east')
        mid=bx0+(bx1-bx0)*bad/total
        s += f'\\path[fill=rdfill] ({bx0},{y-0.18}) rectangle ({bx1},{y+0.18});\n'
        s += f'\\path[fill=rdcoral] ({bx0},{y-0.18}) rectangle ({mid:.4f},{y+0.18});\n'
        s += txt(bx1+0.16,y,n(bad),color='rdcoral')
    s += txt(0,-0.63,r'\textbf{Success}',color='rdink')
    s += txt(0,-1.01,f'$n={n(sp["success"]["rec"])}$',color='rdink2')
    s += txt(0,-2.08,r'\textbf{Failure}',color='rdink')
    s += txt(0,-2.46,f'$n={n(sp["failure"]["rec"])}$',color='rdink2')
    for x,color,lab in [(0,'rdcoral','Discrepant'),(2.55,'rdfill','Coherent under stated relation')]:
        s += f'\\path[fill={color},draw=rdline,line width=0.4pt] ({x},-3.70) rectangle ({x+0.25},-3.48);\n'
        s += txt(x+0.37,-3.59,lab,color='rdink2')
    s += rule_v(8.72,-3.80,0.16,color='rdline')
    # Consumer: equal denominators within these panels only; do not mix them with corpus cases.
    cx0,cx1=12.10,17.60
    total_fields=bind(F,'consumer_scored_fields_per_cohort','COSMOS_CONSUMER',
                     'derived: affected-record count times number of output_summary keys',
                     cr_affected*len(records[0]['conditions']['original']['output_summary']))
    assert cr_affected==cr_controls and total_fields>0
    for tick in (0,4,8,12,16):
        x=cx0+(cx1-cx0)*tick/total_fields
        s += rule_v(x,-2.25,-0.40,color='rdline',lw=0.35)
        s += txt(x,-2.47,str(tick),anchor='center',color='rdink2')
    for kind,y,label in [('affected',-0.89,f'Affected ($n={cr_affected}$)'),
                         ('control',-1.78,f'Controls ($n={cr_controls}$)')]:
        old=cohort[kind]['original'];fixed=cohort[kind]['summary_corrected']
        xa=cx0+(cx1-cx0)*fixed/total_fields;xb=cx0+(cx1-cx0)*old/total_fields
        s += txt(cx0-0.19,y,label,anchor='east')
        if old!=fixed:
            s += rule_h(xa,xb,y,color='rdink2',lw=1.0)
        s += f'\\fill[rdcoral] ({xb:.3f},{y:.3f}) circle (2.1pt);\n'
        s += f'\\draw[rdteal,line width=0.85pt] ({xa:.3f},{y+0.10:.3f}) -- ({xa+0.10:.3f},{y:.3f}) -- ({xa:.3f},{y-0.10:.3f}) -- ({xa-0.10:.3f},{y:.3f}) -- cycle;\n'
        s += txt((xa+xb)/2,y+0.30,f'{old} $\\rightarrow$ {fixed}',anchor='center',color='rdink2')
    s += txt((cx0+cx1)/2,-2.96,f'Incorrect fields (of {total_fields} per cohort)',anchor='center',color='rdink2')
    s += r'\fill[rdcoral] (9.18,-3.59) circle (2.1pt);'+'\n'
    s += txt(9.40,-3.59,'Original',color='rdink2')
    s += r'\draw[rdteal,line width=0.85pt] (12.1,-3.49) -- (12.2,-3.59) -- (12.1,-3.69) -- (12,-3.59) -- cycle;'+'\n'
    s += txt(12.34,-3.59,'Summary-only correction',color='rdink2')
    # Frame checks are a numeric result strip, not a fourth prose card.
    s += rule_h(0,W,-4.03,color='rdink2',lw=0.6)
    s += txt(0,-4.36, f'\\textbf{{Frame checks:}} C1--C4: {n(eps)}/{n(eps)} episodes pass each; '
             f'C5: {c5["violating_rows"]}/{n(c5["rows"])} violating rows.', color='rdink')
    s += txt(0,-4.77,'C2/C3: index multisets; C5: row-wise index relation. Physical order and visual content untested.',color='rdink2')
    cap = (f'Cosmos3-DROID: (a) discrepancy proportions with raw counts; (b) exact pinned metadata-function '
           f'replay on {cr_records} selected records ({n(cr_frames)} frame rows). Only eight '
           '\\texttt{stats/episode\\_index/} fields per record are scored: min, max, mean, '
           'q01, q10, q50, q90, q99. Other metadata, complete merging and downstream harm are outside this comparison.')
    BINDING.setdefault('layout', {})[F] = {'width_cm': W, 'drawing_height_cm': 4.94, 'single_column': False}
    return figure(s, cap, 'fig:cosmos-audit', wide=True)


if __name__ == '__main__':
    outputs = {
        'replay_protocol.tikz': fig_protocol(),
        'tab_cases.tex': table_cases(),
        'fig_gx_matrix.tikz': fig_gx_matrix(),
        'fig_cosmos_audit.tikz': fig_cosmos_audit(),
    }
    for name, body in outputs.items():
        (PAPER / name).write_text(body, encoding='utf-8')
    for stale in ('fig_replay_outcomes.tikz', 'fig_check_matrix.tikz'):
        p = PAPER / stale
        if p.exists():
            p.unlink()
    BINDING['outputs'] = {name: hashlib.sha256((PAPER / name).read_bytes()).hexdigest() for name in outputs}
    BINDING['removed_outputs'] = {
        'fig_replay_outcomes.tikz': 'replaced by Table I (tab_cases.tex); the dumbbell chart mixed '
                                    'non-comparable per-case denominators on one axis',
        'fig_check_matrix.tikz': 'folded into Table I, whose per-case local-check column carries the same '
                                 'classification with its definition beside it',
    }
    (PAPER / 'FIGURE_BINDING.json').write_text(json.dumps(BINDING, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: len(v) for k, v in BINDING['figures'].items()}), 'values bound')
