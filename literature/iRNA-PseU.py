# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import accuracy_score, matthews_corrcoef, cohen_kappa_score, classification_report
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC as YourChosenModel
#
# from src.dataset import load_benchmark_dataset, Species, Modification
# from src.features.encodings import pse_knc
#

def experiment():
    pass

def experiment():
    pass

# encoder = pse_knc.Encoder()
#
# train_data = load_benchmark_dataset(Species.human, Modification.psi)
# test_data = load_benchmark_dataset(Species.human, Modification.psi, True)
#
# train_samples = encoder.fit_transform(train_data.samples).values
# test_samples = encoder.transform(test_data.samples).values
#
# train_targets = train_data.targets
# test_targets = test_data.targets
#
# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
#
#
# def specificity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return tn / (tn + fp)
#
#
# def plot_roc_curve(fpr, tpr, roc_auc, title, filename):
#     plt.figure(figsize=(6, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.savefig(filename, format='pgf', bbox_inches='tight')
#     plt.close()
#
#
# def plot_pr_curve(recall, precision, average_precision, title, filename):
#     plt.figure(figsize=(6, 6))
#     plt.step(recall, precision, color='b', alpha=0.2, where='post')
#     plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title(f'{title}\nAP = {average_precision:.2f}')
#     plt.savefig(filename, format='pgf', bbox_inches='tight')
#     plt.close()
#
#
# def plot_lift_curve(y_true, y_pred_proba, title, filename):
#     percentages = np.arange(0, 110, 10)
#     gains = []
#     for percentage in percentages:
#         threshold = np.percentile(y_pred_proba, 100 - percentage)
#         y_pred = (y_pred_proba >= threshold).astype(int)
#         gains.append(np.sum(y_true[y_pred == 1]) / np.sum(y_true))
#
#     plt.figure(figsize=(6, 6))
#     plt.plot(percentages, gains, marker='o')
#     plt.plot([0, 100], [0, 1], 'k--')
#     plt.xlabel('Percentage of samples')
#     plt.ylabel('Percentage of positive class')
#     plt.title(title)
#     plt.savefig(filename, format='pgf', bbox_inches='tight')
#     plt.close()
#
#
# def plot_cumulative_gain(y_true, y_pred_proba, title, filename):
#     percentages = np.arange(0, 110, 10)
#     gains = []
#     for percentage in percentages:
#         threshold = np.percentile(y_pred_proba, 100 - percentage)
#         y_pred = (y_pred_proba >= threshold).astype(int)
#         gains.append(np.sum(y_true[y_pred == 1]) / np.sum(y_true))
#
#     plt.figure(figsize=(6, 6))
#     plt.plot(percentages, gains, marker='o')
#     plt.plot([0, 100], [0, 1], 'k--')
#     plt.xlabel('Percentage of samples')
#     plt.ylabel('Percentage of positive class')
#     plt.title(title)
#     plt.savefig(filename, format='pgf', bbox_inches='tight')
#     plt.close()
#
#
# # Main code
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# model = YourChosenModel(probability=True)
#
# cv_scores = {
#     'ACC': [], 'MCC': [], 'Specificity': [], 'Cohen Kappa': []
# }
#
# for fold, (train_index, val_index) in enumerate(cv.split(train_samples), 1):
#     X_train, X_val = train_samples[train_index], train_samples[val_index]
#     y_train, y_val = train_targets[train_index], train_targets[val_index]
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)
#     y_pred_proba = model.predict_proba(X_val)[:, 1]
#
#     cv_scores['ACC'].append(accuracy_score(y_val, y_pred))
#     cv_scores['MCC'].append(matthews_corrcoef(y_val, y_pred))
#     cv_scores['Specificity'].append(specificity_score(y_val, y_pred))
#     cv_scores['Cohen Kappa'].append(cohen_kappa_score(y_val, y_pred))
#
#     print(f"Fold {fold} Classification Report:")
#     print(classification_report(y_val, y_pred))
#     print("\n")
#
#     # Generate PGF plots
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     roc_auc = auc(fpr, tpr)
#     plot_roc_curve(fpr, tpr, roc_auc, f'ROC Curve - Fold {fold}', f'roc_curve_fold_{fold}.pgf')
#
#     precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
#     average_precision = average_precision_score(y_val, y_pred_proba)
#     plot_pr_curve(recall, precision, average_precision, f'Precision-Recall Curve - Fold {fold}',
#                   f'pr_curve_fold_{fold}.pgf')
#
#     plot_lift_curve(y_val, y_pred_proba, f'Lift Curve - Fold {fold}', f'lift_curve_fold_{fold}.pgf')
#     plot_cumulative_gain(y_val, y_pred_proba, f'Cumulative Gain Curve - Fold {fold}', f'gain_curve_fold_{fold}.pgf')
#
# # Calculate and print average scores
# print("Average CV Scores:")
# for metric, scores in cv_scores.items():
#     print(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
#
# # Train on whole training set and evaluate on independent test set
# model.fit(train_samples, train_targets)
# test_pred = model.predict(test_samples)
# test_pred_proba = model.predict_proba(test_samples)[:, 1]
#
# print("\nIndependent Test Set Scores:")
# print(f"ACC: {accuracy_score(test_targets, test_pred):.4f}")
# print(f"MCC: {matthews_corrcoef(test_targets, test_pred):.4f}")
# print(f"Specificity: {specificity_score(test_targets, test_pred):.4f}")
# print(f"Cohen Kappa: {cohen_kappa_score(test_targets, test_pred):.4f}")
#
# print("\nTest Set Classification Report:")
# print(classification_report(test_targets, test_pred))
#
# # Generate PGF plots for test set
# fpr, tpr, _ = roc_curve(test_targets, test_pred_proba)
# roc_auc = auc(fpr, tpr)
# plot_roc_curve(fpr, tpr, roc_auc, 'ROC Curve - Independent Test Set', 'roc_curve_test.pgf')
#
# precision, recall, _ = precision_recall_curve(test_targets, test_pred_proba)
# average_precision = average_precision_score(test_targets, test_pred_proba)
# plot_pr_curve(recall, precision, average_precision, 'Precision-Recall Curve - Independent Test Set',
#               'pr_curve_test.pgf')
#
# plot_lift_curve(test_targets, test_pred_proba, 'Lift Curve - Independent Test Set', 'lift_curve_test.pgf')
# plot_cumulative_gain(test_targets, test_pred_proba, 'Cumulative Gain Curve - Independent Test Set',
#                      'gain_curve_test.pgf')
