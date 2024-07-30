import json
from pathlib import Path

from src.utils import get_path

_roc_specie_colors = {
    'h.sapiens': 'darkorange',
    'm.musculus': 'steelblue',
    's.cerevisiae': 'lightgreen',
}

_folds_color = ['purple', 'brown', 'pink', 'cyan', 'yellow']

_specie_name_spaces = {
    'h.sapiens': '~~~',
    'm.musculus': '~~',
    's.cerevisiae': '',
}

def _capitalize_specie_name(specie_name: str):
    return specie_name[0].upper() + '.' + specie_name[2].upper() + specie_name[3:]


def _generate_roc_curve(chunk: str):
    return f'''\\begin{{tikzpicture}}
  \\begin{{axis}}[
    xmin=0,
    ymin=0,
    xmax=1,
    ymax=1,
    clip=false,
    xmajorgrids,
    ymajorgrids,
    tick pos=left,
    legend columns=1,
    tick align=outside,
    grid style={{gridgrey}},
    ylabel={{$Sensitivity$}},
    xlabel={{$1 - Specificity$}},
    ytick style={{color=black}},
    xtick style={{color=black}},
    legend pos=south east,
  ]
{chunk}
    \\addplot[black, dashed, dash pattern=on 4pt off 2pt] table {{
      0 0
      1 1
    }};
  \\end{{axis}}
\\end{{tikzpicture}}
'''

def _generate_specie_roc(data, holder: str):
    chunk = ''
    for item in data:
        part = data[item][holder]

        if part is None:
            continue

        roc_data = part['visualizations']['roc']
        chunk += f'    \\addplot[thick, {_roc_specie_colors[item]}] table {{\n'
        for i in range(0, len(roc_data['fpr'])):
            chunk += f'      {roc_data['fpr'][i]} {roc_data['tpr'][i]}\n'
        chunk += f'    }};\n'
        title = f'\\texttt{{{_capitalize_specie_name(item)}{_specie_name_spaces[item]}}}'

        chunk += f'    \\addlegendentry{{{title} = {'{:.2f}'.format(round(roc_data['auc'][0], 2))}}}\n'

    return _generate_roc_curve(chunk)

def _generate_folds_roc(data):
    chunk = ''
    for j, fold in enumerate(data['folds']):
        roc_data = fold['visualizations']['roc']

        chunk += f'    \\addplot[thick, color={_folds_color[j]}] table {{\n'
        for i in range(0, len(roc_data['fpr'])):
            chunk += f'      {roc_data['fpr'][i]} {roc_data['tpr'][i]}\n'
        chunk += f'    }};\n'
        title = f'\\texttt{{Fold {j}}}'
        chunk += f'    \\addlegendentry{{{title} = {'${:.2f}$'.format(round(roc_data['auc'][0], 2))}}}\n'

    roc_data = data['folds_mean']['visualizations']['roc']
    chunk += f'    \\addplot[ultra thick, color=red] table {{\n'
    for i in range(0, len(roc_data['fpr'])):
        chunk += f'      {roc_data['fpr'][i]} {roc_data['tpr'][i]}\n'
    chunk += f'    }};\n'
    title = f'\\texttt{{Mean~~}}'
    chunk += f'    \\addlegendentry{{{title} = {'${:.2f}$'.format(round(roc_data['auc'][0], 2))}}}\n'

    return _generate_roc_curve(chunk)


def main():
    report_directories = []

    children = Path(get_path('literature')).glob('*')
    for child in children:
        if child.name.startswith('*'):
            continue

        if child.is_dir() and not child.name.startswith('.'):
            report_directories.append(child)

    for report in report_directories:
        report_file = report / 'report.json'

        if not report_file.exists():
            print(f'{report_file} does not exist')
            continue

        print(f'Generating Report {report.name}')

        data = json.loads(open(report_file).read())
        folds_mean_roc = _generate_specie_roc(data, 'folds_mean')
        independent_roc = _generate_specie_roc(data, 'independent')

        output_dir = Path(report / 'out')
        output_dir.mkdir(exist_ok=True)

        open(output_dir / 'folds_mean_roc.tex', 'w').write(folds_mean_roc)
        open(output_dir / 'independent_roc.tex', 'w').write(independent_roc)

        for item in data:
            open(output_dir / f'{item}_folds.tex', 'w').write(_generate_folds_roc(data[item]))



if __name__ == '__main__':
    main()
