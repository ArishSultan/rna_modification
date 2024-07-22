import json
import numpy as np

from src.dataset import load_benchmark_dataset, Species, Modification


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def literature_experiment(name: str, species, op):
    reports = dict()

    for specie in species:
        folds_report, folds_mean, independent = op(
            load_benchmark_dataset(specie, Modification.psi),
            load_benchmark_dataset(specie, Modification.psi, True)
        )

        reports[specie.value] = {
            'folds': list(map(lambda x: x.to_dict(), folds_report)),
            'folds_mean': folds_mean.to_dict(),
            'independent': independent.to_dict() if independent is not None else None,
        }

    with open(f'{name}.json', 'w') as outputfile:
        outputfile.write(json.dumps(reports, cls=_NumpyEncoder))
