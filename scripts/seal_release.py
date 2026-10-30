#!/usr/bin/env python3
"""Record hashes for a deliberately reviewed release; never changes research artifacts."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.preview','__pycache__','.venv','.pytest_cache','.ruff_cache'}
TARGET=ROOT/'provenance/release_files.json'

def main():
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or path==TARGET or any(part in SKIP for part in rel.parts):continue
        if path.suffix in {'.pyc','.pyo'}:continue
        raw=path.read_bytes()
        files.append({'path':rel.as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    TARGET.write_text(json.dumps({'release_date':'2026-10-30','files':files},indent=2)+'\n')
    print(json.dumps({'sealed_files':len(files),'bytes':sum(v['bytes'] for v in files)}))

if __name__=='__main__':main()
