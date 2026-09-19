"""Generate a categorical operation/reference map; no implied chronological pipeline."""
from pathlib import Path


CASES = {
    'RD1': ('Loading', 'Identity / interval'),
    'RD2': ('Migration', 'Identity / interval'),
    'RD3': ('Controller input', 'Convention / frame semantics'),
    'RD4': ('Migration', 'Convention / frame semantics'),
    'RD5': ('Controller input', 'Convention / frame semantics'),
    'RD6': ('Preprocessing', 'Calibration anchor'),
    'RD7': ('Loading', 'Requested-time correspondence'),
    'RD8': ('Statistics', 'Population statistic'),
    'Cosmos': ('Re-release', 'Identity / interval'),
}


def generate(root: Path):
    operations = ['Loading', 'Preprocessing', 'Migration', 'Controller input', 'Statistics', 'Re-release']
    kinds = ['Convention / frame semantics', 'Calibration anchor', 'Requested-time correspondence',
             'Identity / interval', 'Population statistic']
    labels = [r'Convention / frame semantics', 'Calibration anchor', r'Requested-time correspondence',
              'Identity / interval', 'Population statistic']
    marks = ['o', 'triangle', 'square', 'diamond', 'star']
    xs = [4.25 + 2.25*i for i in range(6)]
    def node(x, y, text, extra=''):
        return f'\\node[{extra}] at ({x:.3f},{y:.3f}) {{{text}}};\n'
    body = r'\begin{figure*}[t]'+'\n'+r'\centering'+'\n'
    body += r'\begin{tikzpicture}[x=1cm,y=1cm,font=\rmfamily\footnotesize,inner sep=0pt]'+'\n'
    body += node(8.75, .92, 'Framework and consumer operations')
    body += node(xs[-1], .92, 'Publisher release')
    body += r'\draw[line width=.5pt] (3.18,.65)--(14.33,.65);'+'\n'
    body += r'\draw[line width=.5pt] (14.58,.65)--(16.63,.65);'+'\n'
    for x, op in zip(xs, operations):
        op = {'Controller input': r'Controller-input\\construction', 'Statistics': r'Statistics\\reduction'}.get(op,op)
        body += node(x, .28, op, 'align=center')
    for i,(kind,label,mark) in enumerate(zip(kinds, labels, marks)):
        y = -.44 - i*.53
        body += node(2.98,y,label,'anchor=east')
        body += f'\\draw[black!20,line width=.4pt] (3.18,{y-.26:.3f})--(16.63,{y-.26:.3f});\n'
        for op,x in zip(operations,xs):
            cases = [c for c,v in CASES.items() if v == (op,kind)]
            if cases:
                body += f'\\draw plot[mark={mark},mark size=2.1pt,mark options={{line width=.5pt,fill=white}}] coordinates {{({x-.60:.3f},{y:.3f})}};\n'
                body += node(x-.34,y,', '.join(cases),'anchor=west')
    body += r'\end{tikzpicture}'+'\n'
    body += (r'\caption{Case coverage by operation and required semantic reference. Columns are operation '
             r'categories, not a prescribed execution order; symbols distinguish reference kinds. '
             r'RD3 and RD5 share a broad convention/frame-semantics category but use different rules. '
             r'Cosmos is the complementary re-release audit, outside the eight-case corpus.}'+'\n'
             r'\label{fig:pipeline-coverage}'+'\n'+r'\end{figure*}'+'\n')
    (root/'paper/pipeline_coverage.tikz').write_text(body)
    return {'case_classification': CASES, 'source': 'case-specific operations and references in Table I; Cosmos audit in Section V',
            'classification_scope': 'Descriptive grouping; not an exhaustive taxonomy, chronology, or prevalence claim.',
            'width_cm': 16.63, 'font_size_pt': 8}
