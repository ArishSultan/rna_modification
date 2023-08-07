# from src.data import load_psi
# from src.features import inchi
#
# print(inchi.encode('ACGU'))
#
# # from pathlib import Path
# #
# #
# # def transform_to_csv(in_path: Path, out_path):
# #     data = []
# #     with open(in_path) as file:
# #         lines = file.readlines()
# #
# #         for i in range(0, len(lines), 2):
# #             label = 1 if 'P' in lines[i].split('|')[0] else 0
# #             seq = lines[i + 1].strip()
# #
# #             data.append((label, seq))
# #
# #     with open(out_path, 'w') as file:
# #         for item in data:
# #             file.write(f'{item[0]},{item[1]}\n')
# #
# #
# # transform_to_csv(
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/h.sapiens.csv',
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/h.sapiens.a.csv',
# # )
# #
# # transform_to_csv(
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/m.musculus.csv',
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/m.musculus.a.csv',
# # )
# #
# # transform_to_csv(
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/s.cerevisiae.csv',
# #     '/Users/arish/Workspace/research/rna_modification/data/raw/2ome/training/s.cerevisiae.a.csv',
# # )

import numpy as np

seq1 = np.array([-0.01216619637672269, -0.002069275753486279, 0.0080399133030712, -4.9073733284259874e-05,
                 0.0019833967202388288,
                 0.0019711282869177595, 0.018157281315176053, -0.006113769271664005, -0.012264343843291212,
                 -0.0405430826483458,
                 -0.02236126446652762, 0.001991575675786202, -0.010141904878746983, -3.680529996319404e-05,
                 -0.008117613380771273,
                 0.003991330307119782, -0.002040649409070462, -4.089477773688323e-06, 0.010080562712141661])

# Original
seq2 = np.array([-0.009981382634178918, -0.004739613251224708, 0.011675733839044002, 0.003717590215330791,
                 -0.0009045498355127503, -0.0038888769165718595, 0.006785640947900358, -0.013206848693774802,
                 -0.009804274909829873, -0.04767572739876312, -0.03501846998247005, -0.007921483563288517,
                 -0.008447357267894133,
                 0.0017113188679670557, -0.0017653181461601934, -0.002786969627387642, 0.0022439424823468175,
                 0.0018320741345686778, -0.0034399769537025535])

seq3 = np.array(
    [-0.01216619637672269, -0.016222958328221484, -0.0141904878746984, 0.010092831145462725, 0.010084652189915347,
     -0.010137815400973294, -0.002052917842391526, 0.026233999918210446, 0.02207909050014313, 0.03819572240624873,
     0.011924917188075077, 0.00603606919396393, -0.00812988181409234, -0.008138060769639716, -3.271582218950658e-05,
     0.008044002780844887, -0.006101500838342943, -0.004077209340367233, -4.0894777736881494e-05])

print(seq3 - (seq1 - seq2))
