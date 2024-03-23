import os
import tarfile

from pathlib import Path
from src.utils import get_path
from collections import namedtuple

SpecieInfo = namedtuple("SpecieInfo", ["specie_name", "modification_name"])

SPECIE_NAMES = {
    "hg38": "h.sapiens",
    "mm10": "m.musculus",
    "sacCer3": "s.cerevisiae",
}

MODIFICATION_NAMES = {
    "m6A": "m6a",
    "Pseudo": "psi",
}


def parse_filename(filename):
    parts = filename.split(".")
    specie_name, modification_name = parts[1], parts[2]
    specie_name = SPECIE_NAMES.get(specie_name, specie_name)
    modification_name = MODIFICATION_NAMES.get(modification_name, modification_name)

    return SpecieInfo(specie_name, modification_name)


def process_file(filename: str, tar, member):
    with open(filename, "w") as outfile:
        for line in tar.extractfile(member):
            # Check if line is not empty
            if line.strip():
                parts = line.split(bytes("\t", "utf-8"))
                outfile.write(f"{parts[18].decode("utf-8").replace("T", "U")}, 1\n")

    os.rename(filename, str(filename).replace('.bed', '.csv'))


def prepare_data(root: Path):
    raw_data_dir = root / "dataset" / "raw"
    intermediate_data_dir = root / "dataset" / "intermediate"

    # Create the intermediate directory if it doesn't exist
    intermediate_data_dir.mkdir(exist_ok=True)

    # Get all files in the raw dataset directory
    for filename in os.listdir(raw_data_dir):
        filepath = os.path.join(raw_data_dir, filename)

        # Check if the file is a .tar archive
        if not filename.endswith(".tar"):
            continue

        # Open the tar archive
        with tarfile.open(filepath, "r") as tar:
            # Extract all files ending in .bed to the intermediate directory
            for member in tar.getmembers():
                if member.name.endswith(".bed"):
                    # Parse file name
                    specie_info = parse_filename(member.name)

                    # Create target directory based on modification
                    target_dir = intermediate_data_dir / specie_info.modification_name
                    target_dir.mkdir(exist_ok=True)

                    # Open new file for writing
                    process_file(target_dir / f"{specie_info.specie_name}.bed", tar, member)

    print(f"Successfully extracted .bed files from .tar archives in {raw_data_dir} to {intermediate_data_dir}")


if __name__ == "__main__":
    prepare_data(get_path())
