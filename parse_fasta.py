import os, sys

def parse_fasta(path):
    seqs = {}
    tag, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if tag: seqs[tag] = "".join(buf)
                tag, buf = line[1:].split("|")[0], []
            else:
                buf.append(line)
    if tag: seqs[tag] = "".join(buf)
    return seqs

seqs = parse_fasta(sys.argv[1])

os.makedirs("/data/alignments_multimer", exist_ok=True)

# Write multimer FASTA with stripped headers
with open("/data/input/query_multimer.fasta", "w") as fasta_out:
    for t, s in seqs.items():
        fasta_out.write(f">{t}\n{s}\n")

print(f"Parsed {len(seqs)} sequences: {list(seqs.keys())}")
