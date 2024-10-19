import subprocess
import numpy as np
from pathlib import Path

from ..report import Report


def generate_classification_report(report):
    template = r'''\begin{table}[h!]
  \centering
  \begin{tabular}{rcccc}
    \toprule
    & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} & \textbf{Support} \\
    \midrule

@rows
    \bottomrule
  \end{tabular}
\end{table}'''

    if 'accuracy' in report:
        del report['accuracy']

    rows = []
    for key in report:
        value = report[key]

        rows.append(
            r'    \textbf{%s} & %s & %s & %s & %s \\' % (
                key,
                round(value['precision'], 2),
                round(value['recall'], 2),
                round(value['f1-score'], 2),
                int(value['support'])
            )
        )

    return template.replace('@rows', '\n'.join(rows))


def generate_confusion_matrix(matrix):
    return r'''\begin{table}[h!]
  \centering
  \begin{tabular}{ccc}
    \toprule
     & \textbf{Predicted $+ive$} & \textbf{Predicted $-ive$} \\
    \midrule
    \textbf{Actual $+ive$} & %s & %s \\
    \textbf{Actual $-ive$} & %s & %s \\
    \bottomrule
  \end{tabular}
\end{table}''' % (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1])


def generate_scores_table(scores):
    return r'''\begin{table}[h!]
  \centering
  \begin{tabular}{lc}
    \toprule
    \textbf{Scores} & \textbf{Values} \\
    \midrule
    \texttt{accuracy} & %s \\
    \texttt{mathew's cc} & %s \\
    \texttt{specificity} & %s \\
    \texttt{sensitivity} & %s \\
    \bottomrule
  \end{tabular}
\end{table}''' % (
        round(scores.accuracy, 2),
        round(scores.mcc, 2),
        round(scores.specificity, 2),
        round(scores.sensitivity, 2),
    )


def generate_tikz_plot(xlabel, ylabel, title, plots):
    return r'''\begin{tikzpicture}
  \begin{axis}[
    ylabel={%s},
    xlabel={%s},
    title={%s},
    tick align=outside,
    tick pos=left,
    ytick style={color=black},
    xtick style={color=black},
    grid style={gridgrey},
    xmajorgrids,
    ymajorgrids,
    legend cell align={left},
    legend columns=1,
    legend style={
      fill opacity=0.8,
      draw opacity=1,
      text opacity=1,
      at={(0.97,0.03)},
      anchor=south east,
      draw=legendgray,
    },
  ]
%s
  \end{axis}
\end{tikzpicture}
''' % (ylabel, xlabel, title, plots)


def generate_roc_curve(data):
    fpr = data['fpr']
    tpr = data['tpr']

    entries = []
    for i in range(len(fpr)):
        entries.append(f'      {fpr[i]} {tpr[i]}')

    return generate_tikz_plot(
        'False $+ive$ rate',
        'True $+ive$ rate',
        'Receiver Operating Characteristic (ROC) curve',
        r'''    \addlegendentry{AUC = $%s$}
    \addplot[thick, darkorange] table {
%s
    };
    \addplot[black, dashed, dash pattern=on 4pt off 2pt] table {
      0 0
      1 1
    };''' % (data['auc'].round(decimals=2), '\n'.join(entries))
    )

def generate_average_roc_curve(individual_data, avg_data):
    colors = [
        'steelblue', 'darkorange', 'green', 'gray', 'purple',
        'brown', 'pink', 'cyan', 'magenta', 'yellow'
    ]

    entries = []
    for i, data in enumerate(individual_data):
        fpr = data['fpr']
        tpr = data['tpr']
        color = colors[i % len(colors)]  # Cycle through colors if there are more folds than colors
        entries.append(
            r'''
            \addplot[thin, color=%s, forget plot] table {
            %s
            };
            ''' % (color, '\n'.join([f'      {fpr[j]} {tpr[j]}' for j in range(len(fpr))]))
        )

    # Prepare ROC data for the averaged curve
    avg_fpr = avg_data['fpr']
    avg_tpr = avg_data['tpr']
    avg_auc = avg_data['auc'].round(decimals=2)

    # Add the averaged curve as a bold line
    avg_curve = r'''
    \addlegendentry{Average AUC = $%s$}
    \addplot[very thick, red] table {
          0 0
    %s
    };
    ''' % (avg_auc, '\n'.join([f'      {avg_fpr[i]} {avg_tpr[i]}' for i in range(len(avg_fpr))]))

    # Combine all ROC curves and the average curve into one plot
    return generate_tikz_plot(
        'False $+ive$ rate',
        'True $+ive$ rate',
        'Receiver Operating Characteristic (ROC) curve',
        fr'''
        \addplot[black, dashed, dash pattern=on 4pt off 2pt, forget plot] table {{
          0 0
          1 1
        }};
        
        {'\n'.join(entries) + avg_curve}
        '''
    )


def _generate_gain_curve(data, ylabel, title, is_lift):
    gains1 = data['gains1']
    gains2 = data['gains2']
    percentages = data['percentages']

    plot1 = []
    plot2 = []

    for i in range(len(percentages)):
        plot1.append(f'      {percentages[i]} {gains1[i]}')
        plot2.append(f'      {percentages[i]} {gains2[i]}')

    return generate_tikz_plot(
        'Percentage of Samples',
        ylabel,
        title,
        r'''    \addlegendentry{Class 0}
    \addplot[thick, steelblue] table {
%s
    };

    \addlegendentry{Class 1}
    \addplot[thick, darkorange] table {
%s
    };

    \addlegendentry{Baseline}
    \addplot[black, dashed, dash pattern=on 4pt off 2pt] table {
      0 %s
      1 1
    };''' % ('\n'.join(plot1), '\n'.join(plot2), '1' if is_lift else '0')
    )


def generate_lift_curve(data):
    return _generate_gain_curve(data, 'Lift', 'Lift curve', True)


def generate_pr_curve(data):
    precision = data['precision']
    recall = data['recall']

    filler = []
    entries = []
    for i in range(len(precision)):
        filler.append(f'      --(axis cs:{recall[i]},{precision[i]})')
        entries.append(f'      {recall[i]} {precision[i]}')

    return generate_tikz_plot(
        'Recall',
        'Precision',
        'Precision-Recall curve',
        r'''    \addlegendentry{Precision-Recall curve}

   \addplot[black, opacity=0] table {
      0 0
      1 1
    };
    \addplot [semithick, blue, const plot mark left, opacity=0.3] table {
%s
      1 0
    };
    \addplot [semithick, blue, const plot mark left, opacity=0.1, fill=blue, fill opacity=0.1] table {
%s
    } \closedcycle;''' % (
            '\n'.join(entries),
            '\n'.join(entries),
        )
    )


def generate_cg_curve(data):
    return _generate_gain_curve(data, 'Gain', 'Cumulative gain curve', False)


def generate_latex_report(report: Report, name: str, out: str | Path, generate_pdf=False):
    if type(out) is str:
        out = Path(out)

    out.mkdir(exist_ok=True)

    with open(out / f'{name}.tex', 'w') as file:
        file.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
\usepackage{multirow}
\usepackage{booktabs}
\usepackage[margin=.5in]{geometry}
\usepackage{nopageno}
\usetikzlibrary{patterns}

\definecolor{gridgrey}{RGB}{176,176,176}
\definecolor{legendgray}{RGB}{204,204,204}

\definecolor{steelblue}{RGB}{31,119,180}
\definecolor{darkorange}{RGB}{255,127,14}

\setlength{\paperheight}{12in}

\begin{document}

%s

%s

%s

\begin{table}[h!]
  \centering
  \begin{tabular}{cc}
    %s %s \\
    %s %s
  \end{tabular}
\end{table}

\end{document}
''' % (generate_scores_table(report.scores),
       generate_confusion_matrix(report.tables.confusion_matrix),
       generate_classification_report(report.tables.classification_report),
       generate_roc_curve(report.visualizations.roc),
       generate_pr_curve(report.visualizations.precision_recall),
       generate_lift_curve(report.visualizations.lift),
       generate_cg_curve(report.visualizations.cumulative_gain),
       ))
    if generate_pdf:
        subprocess.run(['tectonic', str(out / f'{name}.tex')])


def generate_kfold_latex_report(reports: list[Report], name: str, out: str | Path, generate_pdf=False):
    if type(out) is str:
        out = Path(out)

    out.mkdir(exist_ok=True)

    generated = []
    for i in range(len(reports)):
        report = reports[i]

        generated.append(r'''
\huge{\textbf{%s}}
\normalsize

%s

%s

%s

\begin{center}
  \begin{tabular}{cc}
    %s %s \\
    %s %s
  \end{tabular}
\end{center}
        ''' % (f'Fold-{i + 1}',
               generate_scores_table(report.scores),
               generate_confusion_matrix(report.tables.confusion_matrix),
               generate_classification_report(report.tables.classification_report),
               generate_roc_curve(report.visualizations.roc),
               generate_pr_curve(report.visualizations.precision_recall),
               generate_lift_curve(report.visualizations.lift),
               generate_cg_curve(report.visualizations.cumulative_gain),
               ))

    with open(out / f'{name}.tex', 'w') as file:
        file.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[margin=.5in]{geometry}
\usepackage{nopageno}
\usetikzlibrary{patterns}

\definecolor{gridgrey}{RGB}{176,176,176}
\definecolor{legendgray}{RGB}{204,204,204}

\definecolor{steelblue}{RGB}{31,119,180}
\definecolor{darkorange}{RGB}{255,127,14}

\setlength{\paperheight}{12in}

\begin{document}

%s

\end{document}
''' % '\n\\clearpage\n'.join(generated))
    if generate_pdf:
        subprocess.run(['tectonic', str(out / f'{name}.tex')])


def average_classification_report(reports):
    avg_report = {}
    num_folds = len(reports)

    for key in reports[0].tables.classification_report:
        avg_report[key] = {
            'precision': np.mean([r.tables.classification_report[key]['precision'] for r in reports]),
            'recall': np.mean([r.tables.classification_report[key]['recall'] for r in reports]),
            'f1-score': np.mean([r.tables.classification_report[key]['f1-score'] for r in reports]),
            'support': int(np.sum([r.tables.classification_report[key]['support'] for r in reports]))
        }

    return avg_report


def average_confusion_matrix(reports):
    matrices = np.array([r.tables.confusion_matrix for r in reports])
    return np.mean(matrices, axis=0).astype(int)


def average_scores(reports):
    return Report.Scores(
        float(np.mean([r.scores.accuracy for r in reports])),
        float(np.mean([r.scores.mcc for r in reports])),
        float(np.mean([r.scores.sensitivity for r in reports])),
        float(np.mean([r.scores.specificity for r in reports]))
    )


def average_roc_curves(reports):
    # Define a common set of FPR values to interpolate
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for r in reports:
        fpr = r.visualizations.roc['fpr']
        tpr = r.visualizations.roc['tpr']
        auc = r.visualizations.roc['auc']

        # Interpolate each TPR to the common FPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs.append(interp_tpr)
        aucs.append(auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)

    return {
        'fpr': mean_fpr,
        'tpr': mean_tpr,
        'auc': mean_auc
    }


def generate_average_kfold_report(reports: list[Report], name: str, out: str | Path, generate_pdf=False):
    if type(out) is str:
        out = Path(out)

    out.mkdir(exist_ok=True)

    avg_classification_report = average_classification_report(reports)
    avg_confusion_matrix = average_confusion_matrix(reports)
    avg_scores = average_scores(reports)
    avg_roc = average_roc_curves(reports)

    individual_roc_data = [report.visualizations.roc for report in reports]

    combined_roc_plot = generate_average_roc_curve(individual_roc_data, avg_roc)

    with open(out / f'{name}.tex', 'w') as file:
        file.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[margin=.5in]{geometry}
\usepackage{nopageno}
\usetikzlibrary{patterns}

\definecolor{gridgrey}{RGB}{176,176,176}
\definecolor{legendgray}{RGB}{204,204,204}

\definecolor{steelblue}{RGB}{31,119,180}
\definecolor{darkorange}{RGB}{255,127,14}

\setlength{\paperheight}{12in}

\begin{document}

\huge{\textbf{Averaged K-Fold Results}}
\normalsize

%s

%s

%s

\begin{center}
  \textbf{Average ROC Curve with K-Folds}
  %s
\end{center}

\end{document}
''' % (generate_scores_table(avg_scores),
       generate_confusion_matrix(avg_confusion_matrix),
       generate_classification_report(avg_classification_report),
       combined_roc_plot
       ))

    if generate_pdf:
        subprocess.run(['tectonic', str(out / f'{name}.tex')])

