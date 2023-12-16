from enum import Enum


class Species(Enum):
    human = 'h.sapiens'
    mouse = 'm.musculus'
    yeast = 's.cerevisiae'

    @staticmethod
    def from_str(value: str):
        match value:
            case 'human':
                return Species.human
            case 'mouse':
                return Species.mouse
            case 'yeast':
                return Species.yeast

        raise ValueError(f"Unknown species, f{value} is not supported yet.")

    @staticmethod
    def all() -> tuple[Enum, Enum, Enum]:
        return Species.human, Species.mouse, Species.yeast
