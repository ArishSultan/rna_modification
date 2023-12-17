from enum import Enum


class Modification(Enum):
    psi = 'psi'
    m6a = 'm6a'
    ome = '2ome'

    @staticmethod
    def from_str(value: str):
        match value:
            case 'psi':
                return Modification.psi
            case 'm6a':
                return Modification.m6a
            case 'ome':
                return Modification.ome

        raise ValueError(f"Unknown modification, f{value} is not supported yet.")

    @staticmethod
    def all() -> tuple[Enum, Enum, Enum]:
        return Modification.psi, Modification.m6a, Modification.ome
