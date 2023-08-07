from src.data import load_psi, Species


def fasta_to_csv():
    pass


def csv_to_fasta():
    dataset = load_psi(Species.human)
    chunk = ''

    for i in range(len(dataset.samples)):
        seq = dataset.samples.iloc[i]["sequence"]
        # seq = seq.replace('U', 'T')
        chunk += f'>Sample_{i}|{dataset.targets.iloc[i]}|training\n'
        chunk += f'{seq}\n'

    with open('sample.txt', 'w') as file:
        file.write(chunk)

csv_to_fasta()
