#!/usr/bin/env python3
"""Convert a3m files to Stockholm format for FastFold multimer pipeline.

Usage:
    python a3m_to_sto.py --inputs file1.a3m file2.a3m
    python a3m_to_sto.py --inputs /dir1/ /dir2/
    python a3m_to_sto.py --inputs /dir1/ file.a3m --output /data/alignments_multimer/
"""

import sys
import os
import re
import argparse


def a3m_to_sto(a3m_path, sto_path):
    entries = []
    with open(a3m_path, 'rb') as f:
        data = f.read().replace(b'\x00', b'').decode()

    name, seq = None, []
    for line in data.splitlines():
        if line.startswith('>'):
            if name is not None:
                entries.append((name, re.sub('[a-z]', '', ''.join(seq))))
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if name is not None:
        entries.append((name, re.sub('[a-z]', '', ''.join(seq))))

    with open(sto_path, 'w') as f:
        f.write('# STOCKHOLM 1.0\n\n')
        for name, sequence in entries:
            f.write(f'{name} {sequence}\n')
        f.write('//\n')

    print(f'Converted {a3m_path} -> {sto_path} ({len(entries)} sequences)')


def collect_a3m_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.endswith('.a3m'):
                    files.append(os.path.join(inp, f))
        elif os.path.isfile(inp) and inp.endswith('.a3m'):
            files.append(inp)
        else:
            print(f'Skipping {inp}: not an .a3m file or directory')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a3m to Stockholm format')
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='a3m files or directories containing a3m files')
    parser.add_argument('--output', default=os.getcwd(),
                        help='Output parent directory (default: current working directory)')
    args = parser.parse_args()

    files = collect_a3m_files(args.inputs)
    if not files:
        print('No .a3m files found')
        sys.exit(1)

    for i, a3m_path in enumerate(files, start=1):
        chain_dir = os.path.join(args.output, str(i))
        os.makedirs(chain_dir, exist_ok=True)
        sto_path = os.path.join(chain_dir, 'uniprot_hits.sto')
        a3m_to_sto(a3m_path, sto_path)
