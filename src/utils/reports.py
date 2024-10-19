from src.utils import get_path
from src.experiment.reports.latex_report import *


def write_reports(report: dict, name: str, modification: str, specie: str):
    main_dir = get_path('dump/reports')
    (main_dir / name / modification / specie).mkdir(exist_ok=True, parents=True)

    generate_kfold_latex_report(report['train'], 'k-folds', main_dir / name / modification / specie, True)
    # generate_latex_report(Report.average_reports(report['train']), 'train', main_dir / name / modification / specie,
    #                       True)
    generate_average_kfold_report(report['train'], 'train', main_dir / name / modification / specie, True)
    if 'test' in report:
        generate_latex_report(report['test'], 'test', main_dir / name / modification / specie, True)
