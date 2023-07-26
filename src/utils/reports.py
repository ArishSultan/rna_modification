from . import get_path
from ..experiment.reports.latex_report import *


def write_reports(report: dict, name: str, modification: str, specie: str):
    main_dir = get_path('reports')

    main_dir.mkdir(exist_ok=True)
    (main_dir / name).mkdir(exist_ok=True)
    (main_dir / name / modification).mkdir(exist_ok=True)
    (main_dir / name / modification / specie).mkdir(exist_ok=True)

    generate_kfold_latex_report(report['train'], 'train', main_dir / name / modification / specie, True)
    if 'test' in report:
        generate_latex_report(report['test'], 'test', main_dir / name / modification / specie, True)
