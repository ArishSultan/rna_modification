import os
import random
import pandas as pd

from pathlib import Path
from src.utils import get_path


def process_data(data_dir: Path):
    # Define output directory
    processed_dir = data_dir / "data" / "processed"
    intermediate_dir = data_dir / "data" / "intermediate"

    # Loop through modification directories (e.g., psi, m6a)
    for modification_dir in os.listdir(intermediate_dir):
        modification_type = modification_dir

        # Skip non-directory entries
        if not os.path.isdir(os.path.join(intermediate_dir, modification_dir)):
            continue

        # Create corresponding directory in processed data
        (processed_dir / modification_dir).mkdir(exist_ok=True)

        for file in os.listdir(intermediate_dir / modification_dir):
            generate_negative_samples(
                intermediate_dir / modification_dir / file,
                modification_type,
                processed_dir / modification_dir / file
            )

    print(f"Processed data and generated negative samples in {processed_dir}")


def generate_negative_samples(file_location, modification_type, out_location):
    # Read existing positive samples
    df_pos = pd.read_csv(file_location, header=None)
    existing_sequences = df_pos[0].tolist()

    # Define central nucleotide based on modification type
    if modification_type == "psi":
        central_nucleotide = "U"
    elif modification_type == "m6a":
        central_nucleotide = "A"
    else:
        raise ValueError(f"Unknown modification type: {modification_type}")

    # Generate negative samples with non-overlapping sequences
    negative_samples = []
    while len(negative_samples) < len(df_pos):
        random_sequence = generate_random_sequence(41, central_nucleotide)
        if random_sequence not in existing_sequences:
            negative_samples.append(random_sequence)

    # Combine positive and negative samples
    combined_samples = pd.concat([df_pos, pd.DataFrame({0: negative_samples, 1: 0})], ignore_index=True)

    # Shuffle and save combined samples
    combined_samples = combined_samples.sample(frac=1)
    combined_samples.to_csv(out_location, header=None, index=False)


def generate_random_sequence(length, central_nucleotide):
    central_index = length // 2
    bases = "ACGU"
    random_bases = [random.choice(bases) for _ in range(length)]
    random_bases[central_index] = central_nucleotide
    return "".join(random_bases)


if __name__ == "__main__":
    process_data(get_path())
