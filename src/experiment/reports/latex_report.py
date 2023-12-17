import subprocess
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
    \texttt{specificity} & %s \\
    \texttt{mathew's cc} & %s \\
    \texttt{cohen Kappa} & %s \\
    \bottomrule
  \end{tabular}
\end{table}''' % (
        round(scores.accuracy, 2),
        round(scores.specificity, 2),
        round(scores.mcc, 2),
        round(scores.cohen_kappa, 2),
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
    };''' % (round(data['auc'], 2), '\n'.join(entries))
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

`\setlength{\paperheight}{12in}

\begin{document}

%s

\end{document}
''' % '\n\\clearpage\n'.join(generated))
    if generate_pdf:
        subprocess.run(['tectonic', str(out / f'{name}.tex')])
