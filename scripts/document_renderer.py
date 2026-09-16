#!/usr/bin/env python3
"""Build review-format PDFs and editable LaTeX from the research Markdown."""
from pathlib import Path
import re
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'

SYMBOLS = {'→':r'\(\to\)', '←':r'\(\leftarrow\)', '≤':r'\(\leq\)', '≥':r'\(\geq\)',
           '≪':r'\(\ll\)', '≠':r'\(\neq\)', '∈':r'\(\in\)', '⊆':r'\(\subseteq\)',
           '√':r'\(\sqrt{}\)', 'ε':r'\(\epsilon\)', 'φ':r'\(\phi\)',
           '∆':r'\(\Delta\)', '∧':r'\(\land\)', '°':r'\(^{\circ}\)'}
SYMBOLS['∎']=r'\(\square\)'
SYMBOLS.update({c:r'\textsuperscript{'+v+'}' for c,v in zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻','0123456789-')})
LINK_PATTERN=r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(([^)]+)\)'

def esc(s):
    s=s.replace('—','---').replace('–','--').replace('−','-').replace('‑','-')
    special={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}','{':r'\{','}':r'\}',
             '~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}
    result=''.join(SYMBOLS.get(c,special.get(c,c)) for c in s)
    result=re.sub(r'[a-fA-F0-9]{30,}',lambda m:r'\allowbreak{}'.join(m[0][i:i+12] for i in range(0,len(m[0]),12)),result)
    for c in '、；。：':result=result.replace(c,c+r'\allowbreak{}')
    result=result.replace(r'\(\to\)',r'\(\to\)\allowbreak{}')
    return result

def urltex(s):
    return s.replace('%',r'\%').replace('#',r'\#')

class Renderer:
    def __init__(self, text, paper):
        self.paper=paper
        self.sources={k:v for k,v in re.findall(r'^\[\^([^]]+)\]:\s*(.+)$',text,re.M)}
        if paper:
            # Cite keys remain stable in prose and generated figures. Bibliography numbers follow
            # first use in the combined source, including table/figure inputs at their insertion point.
            body=text.split('## Sources',1)[0]
            def expand_input(match):
                name='replay_protocol.tikz' if match.group(1) is None else match.group(1).strip()
                path=ROOT/'paper'/name
                if not path.is_file():
                    raise ValueError(f'Missing citation-bearing input: {name}')
                return path.read_text()
            expanded=re.sub(r'<!-- (?:FIGURE_REPLAY_PROTOCOL|LATEX_INPUT:\s*([^>]+?))\s*-->',expand_input,body)
            ordered=[]
            for match in re.finditer(r'\[\^([^]]+)\]|\\cite\{([^}]+)\}',expanded):
                for key in (match.group(1) or match.group(2)).split(','):
                    key=key.strip()
                    if key not in self.sources:
                        raise ValueError(f'Undefined source key: {key}')
                    if key not in ordered:
                        ordered.append(key)
            self.sources={key:self.sources[key] for key in ordered}
            text=body+'## Sources\n\n'+'\n\n'.join(f'[^{k}]: {v}' for k,v in self.sources.items())+'\n'
        self.text=text
        self.numbers={k:i+1 for i,k in enumerate(self.sources)}
        if not paper and all(k.isdigit() for k in self.sources):
            self.numbers={k:int(k) for k in self.sources}
        self.seen=set()
        self.table_count=0

    def inline(self,s,foot=True):
        # Protect generated LaTeX with tokens before escaping the prose.
        tokens=[]
        def save(t):
            tokens.append(t)
            return f'ZZTOKEN{len(tokens)-1}ZZ'
        def cite(m):
            key=m.group(1); n=self.numbers[key]
            if self.paper: return save(r'\cite{'+key+'}')
            if key not in self.seen and foot:
                self.seen.add(key)
                raw=self.sources[key]
                link=re.search(LINK_PATTERN,raw)
                if link:
                    short=f'\\href{{{urltex(link.group(2))}}}{{{esc(link.group(1))}}}. '
                    short+=('完整书目信息见 Sources。' if not self.paper else 'Full citation in Sources.')
                else:
                    short=('完整出处见 Sources。' if not self.paper else 'Full source details in Sources.')
                return save(f'\\footnote[{n}]{{{short}}}')
            return save(f'\\textsuperscript{{\\hyperlink{{source-{n}}}{{{n}}}}}')
        s=re.sub(r'\\\((.*?)\\\)',lambda m:save(m[0]),s)
        s=re.sub(r'\[\^([^]]+)\]',cite,s)
        s=re.sub(LINK_PATTERN,lambda m:save(r'\href{'+urltex(m[2])+'}{'+esc(m[1])+'}'),s)
        def code(m):
            value=esc(m[1])
            # Optional breakpoints for long literal identifiers and paths.
            value=value.replace(r'\_',r'\_\allowbreak{}').replace('/',r'/\allowbreak{}')
            return save(r'\texttt{'+value+'}')
        s=re.sub(r'`([^`]+)`',code,s)
        s=re.sub(r'\*\*([^*]+)\*\*',lambda m:save(r'\textbf{'+esc(m[1])+'}'),s)
        s=re.sub(r'\*([^*]+)\*',lambda m:save(r'\textit{'+esc(m[1])+'}'),s)
        s=esc(s)
        for i,t in enumerate(tokens):s=s.replace(f'ZZTOKEN{i}ZZ',t)
        return s

    def table(self,lines):
        rows=[[c.strip() for c in l.strip().strip('|').split('|')] for l in lines]
        rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',c) for c in r)]
        n=len(rows[0]); assert all(len(r)==n for r in rows),rows
        if rows[0][0]=='Candidate operation':rows[0][0]='Operation'
        means=[sum(min(150,len(r[j])) for r in rows)/len(rows) for j in range(n)]
        weights=[max(9,x)**.7 for x in means]; weights=[x/sum(weights) for x in weights]
        if n==5 and rows[0][0]=='Case': weights=[.065,.235,.205,.18,.315]
        if n==3 and rows[0][0]=='Existing result':weights=[.62,.19,.19]
        self.table_count+=1
        spec='@{}'+''.join(f'p{{{w:.4f}\\dimexpr\\textwidth-{12*(n-1)}pt\\relax}}' for w in weights)+'@{}'
        result=[]
        if self.paper:
            titles={'Validation type':'Validation profiles for the running example.',
                    'Operation':'Supported semantic operations and composition boundaries.',
                    'ID':'Evidence map and analysis units.',
                    'Existing result':'Controlled results with implementation provenance.',
                    'Case':'Replayed RealDefect outcomes and case-specific evidence scope.'}
            titles.update({'Operation group':'Implemented operations and their composition prerequisites.', 'Evidence unit':'Experimental units, executions, and supported interpretations.', 'Omitted prerequisite':'Eight designed counterexamples to omitted validation prerequisites.'})
            titles.update({'Operations':'Registered semantic operations and composition conditions.', 'Evidence':'Experimental evidence units and their interpretation.', 'Split':'Complete pinned Cosmos3-DROID metadata audit; rows are episode records and frames are declared totals.','Rows per artifact':'New-engine scale measurements; median end-to-end seconds from three measured workers.','Public event':'Public prerequisite-omission evidence and bounded execution scope.','Case composition':'Expressed real-case subrelations and remaining full-oracle boundaries.'})
            title=titles.get(rows[0][0],'Comparison of evidence and requirements.')
            if rows[0][0]=='Case' and n==3:title='Executed real-case subrelations and remaining full-oracle conditions.'
            if rows[0][0]=='Case' and n==5: title='Cross-case reference information measured in the replayed scope.'
            if rows[0][0]=='Case' and n==4 and rows[0][1]=='Declared':
                title=('Native Great Expectations 1.6.3 outcomes on paired faulty and reference artifacts. '
                       'A case counts as detected only when the faulty artifact is rejected and the paired '
                       'reference artifact is accepted; \u201cboth rejected\u201d means the suite rejected both '
                       'of this case\u2019s two paired artifacts. Reference artifacts are repaired-branch '
                       'outputs except for RD4, whose reference conversion is constructed, and RD8, whose '
                       'pair is the archived primary condition only. RD6 is a derived portability target '
                       'whose anchor rule applies to 2 of its 73,802 rows.')
            result+=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{'+esc(title)+'}',
                     r'\renewcommand{\arraystretch}{1.18}',r'\begin{tabular}{'+spec+'}',r'\toprule']
        else:
            result+=[r'{\small\renewcommand{\arraystretch}{1.14}',r'\begin{longtable}{'+spec+'}',r'\toprule']
        for i,row in enumerate(rows):
            values=[self.inline(c,False) for c in row]
            if i==0:values=[r'\textbf{'+c+'}' for c in values]
            result+=[' & '.join(values)+r' \\']
            if i==0:
                result+=[r'\midrule']
                if not self.paper:
                    result += [r'\endfirsthead',r'\toprule',
                        ' & '.join(values)+r' \\',r'\midrule',r'\endhead']
        result += [r'\bottomrule',r'\end{tabular}' if self.paper else r'\end{longtable}']
        result += [r'\end{table*}' if self.paper else '}']
        return '\n'.join(result)

    def diagram(self):
        return r'''
\begin{figure*}[t]
\centering
\begin{tikzpicture}[x=1cm,y=1cm,>=stealth]
\node[draw,align=center,font=\small,text width=3.65cm,minimum height=1.40cm] (a) at (0,0) {\textbf{1. Bind}\\Artifacts and transform\\Declarations and evidence};
\node[draw,align=center,font=\small,text width=3.65cm,minimum height=1.40cm] (b) at (4.2,0) {\textbf{2. Decode}\\Storage adapters\\Typed semantic views};
\node[draw,align=center,font=\small,text width=3.65cm,minimum height=1.40cm] (c) at (8.4,0) {\textbf{3. Evaluate}\\Coverage and alignment\\Relational residuals};
\node[draw,align=center,font=\small,text width=3.65cm,minimum height=1.40cm] (d) at (12.6,0) {\textbf{4. Decide}\\Accept / quarantine\\Localized evidence};
\draw[->] (a)--(b);\draw[->] (b)--(c);\draw[->] (c)--(d);
\node[align=center,font=\small] at (6.3,-1.1) {Native readability is checked independently; missing evidence or failed decoding cannot imply acceptance.};
\end{tikzpicture}
\caption{Implemented validation pipeline. Typed operations retain declared prerequisites; only all required passing checks support acceptance.}
\end{figure*}
'''

    def render(self):
        lines=self.text.splitlines(); title=lines[0][2:]; body=[]; i=1; in_abstract=False; in_bib=False
        while i<len(lines):
            l=lines[i].strip()
            if l=='<!-- FIGURE_REPLAY_PROTOCOL -->':
                body.append(r'\input{replay_protocol.tikz}');i+=1;continue
            if l.startswith('<!-- LATEX_INPUT:'):
                body.append(r'\input{'+l.split(':',1)[1].split('-->')[0].strip()+'}');i+=1;continue
            if not l or l.startswith('<!--'):i+=1;continue
            if l==r'\[':
                math=[];i+=1
                while i<len(lines) and lines[i].strip()!=r'\]':math.append(lines[i]);i+=1
                body.append(r'\begin{equation*}'+ '\n'.join(math)+r'\end{equation*}');i+=1;continue
            if re.match(r'^\[\^[^]]+\]:',l):
                m=re.match(r'^\[\^([^]]+)\]:\s*(.*)',l); n=self.numbers[m[1]]
                if self.paper: body.append(r'\bibitem{'+m[1]+'} '+self.inline(m[2],False))
                else: body.append(r'\begin{samepage}\noindent\hypertarget{source-'+str(n)+'}{['+str(n)+']} '+self.inline(m[2],False)+r'\par\smallskip\end{samepage}')
                i+=1;continue
            if l.startswith('## '):
                heading=l[3:]
                if in_abstract:body.append(r'\end{abstract}');in_abstract=False
                if self.paper and heading=='Abstract':
                    body.append(r'\begin{abstract}');in_abstract=True;i+=1;continue
                if heading=='Index Terms':
                    i+=1
                    while i<len(lines) and not lines[i].strip():i+=1
                    terms=[]
                    while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith('#'):
                        terms.append(lines[i].strip());i+=1
                    text=self.inline(' '.join(terms),False)
                    body.append((r'\begin{IEEEkeywords}'+'\n'+text+'\n'+r'\end{IEEEkeywords}')
                                if self.paper else r'\noindent\textbf{Index Terms---}'+text+r'\par')
                    continue
                if heading=='Sources':
                    if self.paper:body += [r'\FloatBarrier',r'\begin{thebibliography}{99}'];in_bib=True
                    else:body += [r'\clearpage',r'\section*{Sources}',r'\small']
                else:
                    heading=re.sub(r'^(?:[IVX]+|[0-9]+)\.\s*','',heading)
                    body.append((r'\section{' if self.paper and heading not in {'Data and Code Availability','AI Assistance Disclosure','Acknowledgments'} else r'\section*{')+self.inline(heading,False)+'}')
                i+=1;continue
            if l.startswith('### '):
                heading=re.sub(r'^[A-Z]\.\s*|^\d+(?:\.\d+)*\s*','',l[4:])
                body.append((r'\subsection{' if self.paper else r'\subsection*{')+self.inline(heading,False)+'}');i+=1;continue
            if l.startswith('|'):
                block=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    block.append(lines[i]);i+=1
                body.append(self.table(block));continue
            if l.startswith('$$'):
                math=l[2:]
                while not math.endswith('$$'):
                    i+=1;math+='\n'+lines[i]
                math=math[:-2]
                if r'\forall' in math and self.paper:
                    # Wide formal definition rendered in two aligned lines.
                    math=math.replace(r'\ \land\ \forall',r'\\ &\land\ \forall')
                    body.append(r'\[\begin{aligned}&'+math+r'\end{aligned}\]')
                else:body.append(r'\['+math+r'\]')
                i+=1;continue
            if l.startswith('```'):
                code=[];i+=1
                while i<len(lines) and not lines[i].startswith('```'):
                    code.append(lines[i]);i+=1
                body+=[r'\begin{quote}\small\ttfamily',r'\par '.join(esc(x) for x in code),r'\end{quote}'];i+=1;continue
            if re.match(r'^(?:\d+\.|-) ',l):
                items=[]
                while i<len(lines) and re.match(r'^(?:\d+\.|-) ',lines[i].strip()):
                    items.append(re.sub(r'^(?:\d+\.|-) ','',lines[i].strip()));i+=1
                body += [r'\begin{itemize}']+[r'\item '+self.inline(x) for x in items]+[r'\end{itemize}'];continue
            paragraph=[l];i+=1
            while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||\[\^|\$\$|```|<!--)',lines[i].strip()):
                paragraph.append(lines[i].strip());i+=1
            p=' '.join(paragraph)
            if p.startswith('**Figure 1. Four-stage validation pipeline.'):
                body.append(self.diagram())
            else:body.append(self.inline(p)+r'\par')
        if in_abstract:body.append(r'\end{abstract}')
        if in_bib:body.append(r'\end{thebibliography}')
        return self.preamble(title)+'\n\n'.join(body)+'\n\\end{document}\n'


    def ieee_preamble(self,title):
        return r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,booktabs,array,longtable,tabularx,xurl,tikz,graphicx}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{placeins}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{1em}
% Float placement policy only: no change to type size, leading, column width or margins.
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.60}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\dblfloatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{3}
\setcounter{totalnumber}{4}
\begin{document}
\title{"""+esc(title)+r"""}
\maketitle
"""

    def preamble(self,title):
        if self.paper: return self.ieee_preamble(title)
        mode='10pt,twocolumn' if self.paper else '11pt'
        fonts=(r'''
\setmainfont[Path=/System/Library/Fonts/Supplemental/,BoldFont={Times New Roman Bold.ttf},ItalicFont={Times New Roman Italic.ttf},BoldItalicFont={Times New Roman Bold Italic.ttf}]{Times New Roman.ttf}
\setsansfont{Arial}
\setmonofont[Scale=0.87]{Courier New}
''' if self.paper else r'''
\setmainfont[Path=/Library/Fonts/,AutoFakeBold=2,AutoFakeSlant=0.18]{Arial Unicode.ttf}
\setsansfont[Path=/Library/Fonts/,AutoFakeBold=2,AutoFakeSlant=0.18]{Arial Unicode.ttf}
\setmonofont[Path=/Library/Fonts/,Scale=0.88]{Arial Unicode.ttf}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
''')
        return rf'''\documentclass[{mode},letterpaper]{{article}}
\usepackage[left=0.70in,right=0.70in,top=0.67in,bottom=0.72in]{{geometry}}
\usepackage{{fontspec}}
{fonts}
\usepackage{{amsmath,amssymb,booktabs,array,longtable,tabularx,xurl,tikz}}
\usepackage{{placeins}}
\usepackage[unicode,hidelinks]{{hyperref}}
\hypersetup{{pdftitle={{{esc(title)}}},pdfauthor={{}},pdfsubject={{Research draft; reproduced and new-system evidence distinguished}}}}
\setlength{{\columnsep}}{{0.27in}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{{4 if self.paper else 2}pt plus 1pt minus 1pt}}
\setlength{{\emergencystretch}}{{2em}}
\setlength{{\skip\footins}}{{10pt}}
\renewcommand{{\arraystretch}}{{1.1}}
\makeatletter
\renewcommand\section{{\@startsection{{section}}{{1}}{{0pt}}{{10pt}}{{5pt}}{{\normalfont\large\bfseries}}}}
\renewcommand\subsection{{\@startsection{{subsection}}{{2}}{{0pt}}{{7pt}}{{4pt}}{{\normalfont\normalsize\bfseries}}}}
\makeatother
\begin{{document}}
'''+((r'\twocolumn[\begin{center}{\LARGE\bfseries '+esc(title)+r'}\end{center}\vspace{6pt}]') if self.paper else
       r'{\LARGE\bfseries '+esc(title)+r'\par}\vspace{10pt}')+'\n'

def build(path,paper):
    text=path.read_text()
    tex=path.with_suffix('.tex');tex.write_text(Renderer(text,paper).render())
    builddir=ROOT/'tmp'/'latex'/path.stem;builddir.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        p=subprocess.run(['pdflatex' if paper else 'xelatex','-interaction=nonstopmode','-halt-on-error',
             f'-output-directory={builddir}',str(tex)],cwd=path.parent,text=True,capture_output=True)
        (builddir/f'run{attempt}.txt').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout[-4500:])
    target=path.with_suffix('.pdf');target.write_bytes((builddir/(path.stem+'.pdf')).read_bytes())
    log=(builddir/(path.stem+'.log')).read_text()
    warnings=[x for x in log.splitlines() if 'Overfull' in x or 'Missing character' in x or 'undefined' in x]
    return {'pdf':str(target),'warnings':warnings}

if __name__=='__main__':
    jobs=[(OUT/'papers/Paper_A_RealDefect_R.md',True),(OUT/'papers/Paper_B_RoboContract_ICDE.md',True),
          (OUT/'research/RoboContract_Reconstruction_Report_CN.md',False)]
    result=[build(*j) for j in jobs]
    (OUT/'build_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
