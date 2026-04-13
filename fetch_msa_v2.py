import requests, time, os, sys

def parse_fasta(path):
    seqs = {}
    tag, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if tag: seqs[tag] = "".join(buf)
                tag, buf = line[1:], []
            else:
                buf.append(line)
    if tag: seqs[tag] = "".join(buf)
    return seqs

seqs = parse_fasta(sys.argv[1])

os.makedirs("/data/alignments_multimer", exist_ok=True)

for tag, seq in seqs.items():
    r = requests.post("https://api.colabfold.com/ticket/msa",
                      data={"q": f">{tag}\n{seq}", "mode": "all"})
    print(f"{tag}: HTTP {r.status_code} | {r.text[:200]}")
    resp = r.json()
    if "id" not in resp:
        raise RuntimeError(f"{tag} submission failed: {resp.get('status')} - {resp.get('reason', '')}")
    ticket = resp["id"]
    print(f"{tag}: ticket {ticket}")

    while True:
        status = requests.get(f"https://api.colabfold.com/ticket/{ticket}").json()["status"]
        if status == "COMPLETE": break
        if status == "ERROR": raise RuntimeError(f"{tag} failed")
        time.sleep(15)

    result = requests.get(f"https://api.colabfold.com/result/download/{ticket}").content
    import tarfile, io
    with tarfile.open(fileobj=io.BytesIO(result)) as tar:
        f = tar.extractfile("uniref.a3m")
        cleaned = f.read().replace(b"\x00", b"")

    os.makedirs(f"/data/alignments_multimer/{tag}", exist_ok=True)
    with open(f"/data/alignments_multimer/{tag}/uniref.a3m", "wb") as out:
        out.write(cleaned)
    print(f"{tag}: done")
