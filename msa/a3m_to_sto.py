import re

def a3m_to_sto(a3m_path, sto_path, query_seq):
    entries = []
    with open(a3m_path, 'rb') as f:
        data = f.read().replace(b'\x00', b'').decode()
    
    name, seq = None, []
    for line in data.splitlines():
        if line.startswith('>'):
            if name and seq:
                entries.append((name, re.sub('[a-z]', '', ''.join(seq))))
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if name and seq:
        entries.append((name, re.sub('[a-z]', '', ''.join(seq))))

    with open(sto_path, 'w') as f:
        f.write('# STOCKHOLM 1.0\n\n')
        for name, seq in entries:
            f.write(f'{name} {seq}\n')
        f.write('//\n')
    print(f'Written {len(entries)} sequences to {sto_path}')

# Chain A
a3m_to_sto(
    '/data/alignments_multimer/1A6U_A/uniref.a3m',
    '/data/alignments_multimer/1A6U_A/uniprot_hits.sto',
    'AVVTQESALTTSPGETVTLTCRSSTGAVTTSNYANWVQEKPDHLFTGLIGGTNNRAPGVPARFSGSLIGNKAALTITGAQTEDEAIYFCALWYSNHWVFGGGTKLTVL'
)

# Chain B
a3m_to_sto(
    '/data/alignments_multimer/1A6U_B/uniref.a3m',
    '/data/alignments_multimer/1A6U_B/uniprot_hits.sto',
    'QVQLQQPGAELVKPGASVKLSCKASGYTFTSYWMHWVKQRPGRGLEWIGRIDPNSGGTKYNEKFKSKATLTVDKPSSTAYMQLSSLTSEDSAVYYCARYDYYGSSYFDYWGQGTTVTVSS'
)
