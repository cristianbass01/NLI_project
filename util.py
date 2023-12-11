from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, coverage_error, label_ranking_average_precision_score, label_ranking_loss
import numpy as np
import pandas as pd

def generate_metrics_latex_table(model_name, task_number, true_labels, binary_predictions, prediction_probs, target_names):
    report = classification_report(true_labels, binary_predictions, target_names=target_names, digits=3, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    df = df.rename({'precision': r'\textbf{Precision}', 'recall': r'\textbf{Recall}', 'f1-score': r'\textbf{F1-Score}', 'support': r'\textbf{Support}'}, axis=1)

    # Generating additional metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision, recall, f_score, _ = precision_recall_fscore_support(true_labels, binary_predictions, average='micro')

    # Calculating multilabel-specific metrics
    coverage_err = coverage_error(true_labels, prediction_probs)
    lrap = label_ranking_average_precision_score(true_labels, prediction_probs)
    ranking_loss = label_ranking_loss(true_labels, prediction_probs)

    # Calculate best/worst/expected values where applicable
    # Best possible coverage error is the average number of true labels per instance
    best_coverage = true_labels.sum(axis=1).mean()
    # The worst case is the total number of labels
    worst_coverage = true_labels.shape[1]

    # For LRAP, the best value is 1 and the worst is 0. Expected is the baseline or random performance.
    best_lrap = 1.0
    worst_lrap = 0.0  # This is theoretical; in practice, it's unlikely to get 0

    # For ranking loss, the best value is 0. 
    best_rl = 0.0
    # The worst case needs to account for the number of possible incorrect pairings. For each instance, it's the number of true labels times the number of false labels
    worst_rl = np.mean([(sum(row) * (len(row) - sum(row))) for row in true_labels])


    # Converting to LaTeX table
    latex_table = df.to_latex(float_format="%.3f", column_format='|l|c|c|c|c|')
    # Removing some stuff from df.to_latex() output
    latex_table = latex_table.replace('\\toprule\n ', r'\hline' + '\n' + r'\textbf{Class}') \
                             .replace('\\midrule\n', '') \
                             .replace('\\bottomrule', r'\multicolumn{5}{c}{}\\') \
                             .replace('\\end{tabular}\n', '') \
                             .replace(r'\\', r'\\ \hline') \
                             .replace('\nmicro avg','\\hline\nmicro avg')
    
    # Adding overall metrics
    overall_metrics = f"""
{latex_table}
\\textbf{{Accuracy}}                    & \\multicolumn{{4}}{{c|}}{{{accuracy:.3f}}}                                 \\\\ \\hline
\\textbf{{Overall Precision}}           & \\multicolumn{{4}}{{c|}}{{{precision:.3f}}}                                \\\\ \\hline
\\textbf{{Overall Recall}}              & \\multicolumn{{4}}{{c|}}{{{recall:.3f}}}                                   \\\\ \\hline
\\textbf{{Overall F1-Score}}            & \\multicolumn{{4}}{{c|}}{{{f_score:.3f}}}                                  \\\\ \\hline
\\textbf{{Label Ranking Avg Precision}} & \\multicolumn{{4}}{{c|}}{{{lrap:.3f}}}                                    \\\\ \\hline
\\textbf{{Coverage Error}}              & \\multicolumn{{4}}{{c|}}{{{coverage_err:.3f} (worst: {worst_coverage:.3f}, best: {best_coverage:.3f})}}                             \\\\ \\hline
\\textbf{{Ranking Loss}}                & \\multicolumn{{4}}{{c|}}{{{ranking_loss:.3f} (worst: {worst_rl:.3f}, best: {best_rl:.3f})}}                             \\\\ \\hline
\\end{{tabular}}
"""

    # Final LaTeX output with caption and label
    final_latex_output = f"""
\\begin{{table}}[h]
\\centering
{overall_metrics}
\\caption{{Metrics Overview of {model_name} Model for Task {task_number}}}
\\label{{table:{model_name}_metrics_task_{task_number}}}
\\end{{table}}
    """

    # Print or write to a file
    with open('metrics.tex', 'w') as f:
        f.write(final_latex_output)
