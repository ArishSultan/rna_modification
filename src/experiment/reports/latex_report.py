from pathlib import Path
from ..report import Report


def generate_classification_report(report):
    template = r'''\begin{table}[h]
  \centering
  \begin{tabular}{|r|c|c|c|c|}
    \hline
    & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} & \textbf{Support} \\
    \hline

@rows
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
                round(value['support'], 2)
            )
        )
        rows.append(r'    \hline')

    return template.replace('@rows', '\n'.join(rows))


def generate_confusion_matrix(matrix):
    return r'''\begin{table}[h]
  \centering
  \begin{tabular}{|c|c|c|}
    \hline
     & \textbf{Predicted $+ive$} & \textbf{Predicted $-ive$} \\
    \hline
    \textbf{Actual $+ive$} & %s & %s \\
    \hline
    \textbf{Actual $-ive$} & %s & %s \\
    \hline
  \end{tabular}
\end{table}''' % (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1])


def generate_scores_table(scores):
    return r'''\begin{table}[h]
  \centering
  \begin{tabular}{|c|c|}
    \hline
    \textbf{Scores} & \textbf{Values} \\
    \hline
    Accuracy & %s \\
    \hline
    Specificity & %s \\
    \hline
    MCC & %s \\
    \hline
    Cohen Kappa & %s \\
    \hline
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


def _generate_gain_curve(data, ylabel, title):
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
    \addplot[black, line width=1pt, dashed, dash pattern=on 4pt off 2pt] table {
      0 1
      1 1
    };''' % ('\n'.join(plot1), '\n'.join(plot2))
    )


def generate_lift_curve(data):
    return _generate_gain_curve(data, 'Lift', 'Lift curve')


def generate_pr_curve(data):
    precision = data['precision']
    recall = data['recall']

    entries = []
    for i in range(len(precision)):
        entries.append(f'      {recall[i]} {precision[i]}')

    return generate_tikz_plot(
        'Recall',
        'Precision',
        'Precision-Recall curve',
        r'''    \addlegendentry{Precision-Recall curve}
   \addplot[thick, steelblue] table {
%s
   };''' % '\n'.join(entries)
    )


def generate_cg_curve(data):
    return _generate_gain_curve(data, 'Gain', 'Cumulative gain curve')


def generate_latex_report(report: Report, name: str, out: Path):
    out.mkdir(exist_ok=True)

    with open(out / f'{name}.tex', 'w') as file:
        file.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
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

%s

%s

\begin{table}[h!]
  \centering
  \begin{tabular}{cc}
    %s %s \\
    \hline
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
       generate_lift_curve(report.visualizations.cumulative_gain),
       ))


def generate_kfold_latex_report(reports: list[Report], name: str, out: Path):
    out.mkdir(exist_ok=True)

    generated = []
    for i in range(len(reports)):
        report = reports[i]
        print(report)

        generated.append(r'''
\begin{center}
\huge{\textbf{%s}}
\end{center}

%s

%s

%s

\begin{table}[h!]
  \centering
  \begin{tabular}{cc}
    %s %s \\
    \hline
    %s %s
  \end{tabular}
\end{table}
        ''' % (f'{i + 1} Fold',
               generate_scores_table(report.scores),
               generate_confusion_matrix(report.tables.confusion_matrix),
               generate_classification_report(report.tables.classification_report),
               generate_roc_curve(report.visualizations.roc),
               generate_pr_curve(report.visualizations.precision_recall),
               generate_lift_curve(report.visualizations.lift),
               generate_lift_curve(report.visualizations.cumulative_gain),
               ))

    with open(out / f'{name}.tex', 'w') as file:
        file.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
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
''' % '\n'.join(generated))
