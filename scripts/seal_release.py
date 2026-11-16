#!/usr/bin/env python3
"""Record hashes for a deliberately reviewed release; never changes research artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'provenance/release_files.json'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-date',required=True,help='Reviewed release date, YYYY-MM-DD')
    args=parser.parse_args()
    files=[]
    # Only intentionally tracked/staged files belong to the publication inventory.
    # Never sweep private notes, untracked drafts, credentials, or local caches.
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in sorted(set(names)-{''}):
        path=ROOT/name
        if path==TARGET:continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Publication file is missing or a symlink: {name}')
        raw=path.read_bytes()
        files.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    TARGET.write_text(json.dumps({'release_date':args.release_date,
                                 'scope':'Reviewed tracked files only; private and untracked work excluded',
                                 'files':files},indent=2)+'\n')
    print(json.dumps({'sealed_files':len(files),'bytes':sum(v['bytes'] for v in files)}))

if __name__=='__main__':main()
