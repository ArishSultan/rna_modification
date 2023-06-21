from pathlib import Path


def _make_point_pair(arr1, arr2):
    buffer = ''
    for i in range(len(arr1)):
        buffer += f'({arr1[i]}, {arr2[i]})'
    return buffer


def plot_roc(roc_data: dict, title='ROC Curve', output_path: Path = 'roc.tex'):
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc = roc_data['auc']

    chunk = fr'''
\begin{{tikzpicture}}
  \begin{{axis}}[
    grid=major,
    xmin=0, xmax=1,
    ymin=0, ymax=1,
    title={{{title}}},
    ylabel={{True Positive Rate}},
    xlabel={{False Positive Rate}},
    xtick={{0, 0.2, 0.4, 0.6, 0.8, 1}},
    ytick={{0, 0.2, 0.4, 0.6, 0.8, 1}},
  ]
    \addplot[blue, mark=none] coordinates {{{_make_point_pair(fpr, tpr)}}};

    \addplot[dashed, gray] coordinates {{(0, 0) (1, 1)}};
    \addlegendentry{{AUC $= {auc}$}}
  \end{{axis}}
\end{{tikzpicture}}
'''

    with open(output_path, 'w') as fig_file:
        fig_file.write(chunk)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    # plt.title(f'For {key.capitalize()}')
    # plt.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    # figures.append(rf'\input{{{filename}}}')
    # tikz.save(out_dir / filename)
    # plt.clf()
