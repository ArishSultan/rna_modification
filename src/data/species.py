from enum import Enum


class Species(Enum):
    """
    Enum representing different species.

    The Species enum provides a set of predefined species with their corresponding codes.
    Each species is represented as a member of the Enum class, where the member name is
    the common name of the species, and the value is the corresponding code.

    Attributes:
        human (str): Code for the human species ('h.sapiens').
        mouse (str): Code for the mouse species ('m.musculus').
        yeast (str): Code for the yeast species ('s.cerevisiae').
    """
    human = 'h.sapiens'
    mouse = 'm.musculus'
    yeast = 's.cerevisiae'

    @staticmethod
    def all() -> tuple[Enum, Enum, Enum]:
        """
        Get all species defined in the Species enum.

        Returns:
            tuple[Enum]: A tuple containing all species defined in the Species enum.
        """
        return Species.human, Species.mouse, Species.yeast
