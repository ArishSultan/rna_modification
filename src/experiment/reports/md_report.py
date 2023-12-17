import json
from pathlib import Path
from ..report import Report

import matplotlib.pyplot as plt


def generate_md_report(report: Report, name: Path, assets_dir: Path | str = None):
    if name.exists():
        with open(name / 'README.md', 'w') as file:
            file.write(json.dumps(report.to_json()))

    if assets_dir is None:
        assets_dir = name / 'assets'

    assets_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(report.visualizations.roc, assets_dir / 'roc.png')
    plot_lift_curve(report.visualizations.lift, assets_dir / 'lift.png')
    plot_pr_curve(report.visualizations.precision_recall, assets_dir / 'precision_recall.png')
    plot_lift_curve(report.visualizations.cumulative_gain, assets_dir / 'cumulative_gain.png')


def plot_roc_curve(data, destination: Path | str):
    plt.clf()
    plt.plot(data['fpr'], data['tpr'], color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(destination, format='png', bbox_inches='tight')


def plot_pr_curve(data, destination: Path | str):
    plt.clf()
    plt.plot(data['recall'], data['precision'], color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(destination, format='png', bbox_inches='tight')


def plot_lift_curve(data, destination: Path | str):
    plt.clf()
    plt.plot(data['percentages'], data['gains1'], lw=2, label='Class {}'.format(0))
    plt.plot(data['percentages'], data['gains2'], lw=2, label='Class {}'.format(1))
    plt.plot([0, 1], [1, 1], 'k--', lw=1, label='Baseline')
    plt.xlabel('Percentage of sample')
    plt.ylabel('Lift')
    plt.grid('on')
    plt.legend(loc='lower right')
    plt.savefig(destination, format='png', bbox_inches='tight')


def plot_cg_curve(data, destination: Path | str):
    plt.clf()
    plt.plot(data['percentages'], data['gains1'], lw=2, label='Class {}'.format(0))
    plt.plot(data['percentages'], data['gains2'], lw=2, label='Class {}'.format(1))
    plt.plot([0, 1], [1, 1], 'k--', lw=1, label='Baseline')
    plt.xlabel('Percentage of sample')
    plt.ylabel('Gain')
    plt.grid('on')
    plt.legend(loc='lower right')
    plt.savefig(destination, format='png', bbox_inches='tight')
