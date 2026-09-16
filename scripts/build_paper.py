#!/usr/bin/env python3
"""Build the project manuscript with pdfLaTeX and the bundled IEEEtran conference class."""
import json
from document_renderer import ROOT,build
if __name__ == '__main__':
    result=build(ROOT/'paper/Paper_A_RealDefect_R.md',True)
    (ROOT/'paper/build_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
