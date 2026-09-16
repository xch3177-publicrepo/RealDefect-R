#!/usr/bin/env python3
"""Verify the exact release snapshot before any experiment creates mutable files."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'MANIFEST.json').read_text());bad=[]
for x in m['files']:
 p=ROOT/x['path']
 if not p.is_file() or p.stat().st_size!=x['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:bad.append(x['path'])
print(json.dumps({'verified_files':len(m['files']),'mismatches':bad},indent=2))
raise SystemExit(bool(bad))
