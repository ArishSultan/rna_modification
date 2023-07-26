from src.model import random_forest
from src.experiment import Experiment
from src.data import load_psi, load_2ome, Species
from src.features.encodings import pstnpss

from src.utils import write_reports

human_encoder = pstnpss.Encoder(Species.human)
mouse_encoder = pstnpss.Encoder(Species.mouse)
yeast_encoder = pstnpss.Encoder(Species.yeast)

model = random_forest.Factory()

human_psi = Experiment(
    factory=model,
    encoding=human_encoder,
    test=load_psi(Species.human, True),
    train=load_psi(Species.human, False),
).run()

mouse_psi = Experiment(
    test=None,
    factory=model,
    encoding=mouse_encoder,
    train=load_psi(Species.mouse, False),
).run()

yeast_psi = Experiment(
    factory=model,
    encoding=yeast_encoder,
    test=load_psi(Species.yeast, True),
    train=load_psi(Species.yeast, False),
).run()

# human_2ome = Experiment(
#     factory=model,
#     encoding=human_encoder,
#     test=load_2ome(Species.human, True),
#     train=load_2ome(Species.human, False),
# ).run()
#
# mouse_2ome = Experiment(
#     test=None,
#     factory=model,
#     encoding=mouse_encoder,
#     train=load_2ome(Species.mouse, False),
# ).run()
#
# yeast_2ome = Experiment(
#     test=None,
#     factory=model,
#     encoding=yeast_encoder,
#     train=load_2ome(Species.yeast, False),
# ).run()

write_reports(human_psi, 'rf_pstnpss', 'psi', 'human')
write_reports(mouse_psi, 'rf_pstnpss', 'psi', 'mouse')
write_reports(yeast_psi, 'rf_pstnpss', 'psi', 'yeast')

# write_reports(human_2ome, 'rf_pstnpss', '2ome', 'human')
# write_reports(mouse_2ome, 'rf_pstnpss', '2ome', 'mouse')
# write_reports(yeast_2ome, 'rf_pstnpss', '2ome', 'yeast')
