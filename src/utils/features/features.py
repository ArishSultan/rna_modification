import pickle
from src.utils import get_path

_METHODS_DICT = {'DAC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'DCC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'DACC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TAC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TCC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TACC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PseKNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PCPseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PCPseTNC': [],
                 'SCPseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'SCPseTNC': []}

_DATA_FILE_DICT = {'DAC': 'dirnaPhyche.data',
                   'DCC': 'dirnaPhyche.data',
                   'DACC': 'dirnaPhyche.data',
                   'TAC': 'dirnaPhyche.data',
                   'TCC': 'dirnaPhyche.data',
                   'TACC': 'dirnaPhyche.data',
                   'PseDNC': 'dirnaPhyche.data',
                   'PseKNC': 'dirnaPhyche.data',
                   'PCPseDNC': 'dirnaPhyche.data',
                   'PCPseTNC': '',
                   'SCPseDNC': 'dirnaPhyche.data',
                   'SCPseTNC': ''}


def get_info_file(method: str):
    filename = _DATA_FILE_DICT[method]
    with open(get_path(f'data/raw/features/{filename}'), 'rb') as file:
        data = pickle.load(file)
    return _METHODS_DICT[method], data
