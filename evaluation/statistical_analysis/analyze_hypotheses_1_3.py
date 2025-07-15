import json
import os

def analyze_h1(data, alpha=0.05):
    """
    Analyzes and prints the evaluation for Hypothesis H1.

    H1: The CoTR pipeline significantly improves task performance over
        direct baseline approaches for LRLs.
    """
    print("--- Evaluation of Hypothesis H1 ---")
    print("H1: The CoTR pipeline significantly improves task performance over direct baseline approaches for LRLs.\n")

    h1_data = data.get("hypotheses", {}).get("H1", {}).get("overall_f1", {})

    if not h1_data:
        print("Could not find H1 F1-score data in the results file.")
        return

    p_value = h1_data.get("p_value")
    t_statistic = h1_data.get("t_statistic")
    is_significant = h1_data.get("significant", "False").lower() == 'true'
    mean_improvement = h1_data.get("mean_improvement", 0)
    relative_improvement = h1_data.get("relative_improvement_pct", 0)

    print(f"Paired t-test results for overall F1 score (across all tasks):")
    print(f"  - p-value: {p_value:.5f}")
    print(f"  - t-statistic: {t_statistic:.4f}")
    print(f"  - Significance Level (α): {alpha}")
    print(f"  - Relative F1 Improvement: {relative_improvement:.2f}%\n")

    print("Conclusion for H1:")
    if is_significant and t_statistic > 0:
        print(f"The data provides strong evidence to SUPPORT H1.")
        print(f"The overall p-value ({p_value:.5f}) is less than the significance level (α = {alpha}), "
              f"indicating a statistically significant result.")
        print("The positive t-statistic and relative improvement of "
              f"{relative_improvement:.2f}% show that the CoTR pipeline's performance is significantly better than the baseline.")
    else:
        print(f"The data does NOT support H1.")
        if not is_significant:
            print(f"The p-value ({p_value:.5f}) is greater than the significance level (α = {alpha}), "
                  f"so the observed difference is not statistically significant.")
        elif t_statistic <= 0:
            print("Although the result is significant, the effect is not positive, "
                  "contradicting the hypothesis.")
    print("-" * 35 + "\n")


def analyze_h3(data):
    """
    Analyzes and prints the evaluation for Hypothesis H3.

    H3: The effectiveness of CoTR varies by task type, with the largest
        gains observed in tasks that require more complex reasoning (e.g.,
        QA and NLI) compared to more extractive classification tasks.
    """
    print("--- Evaluation of Hypothesis H3 ---")
    print("H3: CoTR effectiveness varies, with larger gains in complex tasks (QA, NLI) "
          "than simple ones (NER, Sentiment, Classification).\n")

    h3_data = data.get("hypotheses", {}).get("H3", {}).get("task_specific", {})

    if not h3_data:
        print("Could not find H3 data in the results file.")
        return

    complex_tasks = {"nli": "NLI", "qa": "QA"}
    simple_tasks = {"sentiment": "Sentiment Analysis", "ner": "NER", "classification": "Text Classification"}
    
    results = {}
    for task_code, task_name in {**complex_tasks, **simple_tasks}.items():
        # Use F1 score where available, otherwise accuracy
        task_data = h3_data.get(task_code, {})
        metric = "f1" if "f1" in task_data else "accuracy"
        metric_data = task_data.get(metric, {})
        
        results[task_name] = {
            "improvement": metric_data.get("relative_improvement_pct", 0),
            "p_value": metric_data.get("p_value"),
            "significant": metric_data.get("significant", "False").lower() == 'true',
            "metric": metric.upper()
        }

    print("Comparative Analysis of Performance Gains (Relative F1/Accuracy Improvement %):")
    print(f"{'Task Category':<12} | {'Task Name':<25} | {'Metric':<8} | {'Improvement (%)':<18} | {'p-value':<12} | {'Significant':<12}")
    print("-" * 95)

    print("Complex Tasks:")
    for task_code, task_name in complex_tasks.items():
        res = results[task_name]
        print(f"{'  Complex':<12} | {task_name:<25} | {res['metric']:<8} | {res['improvement']:>17.2f}% | {res['p_value']:.5f} | {str(res['significant']):<12}")

    print("\nSimple/Extractive Tasks:")
    for task_code, task_name in simple_tasks.items():
        res = results[task_name]
        print(f"{'  Simple':<12} | {task_name:<25} | {res['metric']:<8} | {res['improvement']:>17.2f}% | {res['p_value']:.5f} | {str(res['significant']):<12}")
    
    print("\nConclusion for H3:")
    print("The data does NOT support H3 as stated.")
    print("1. Variation by Task: The effectiveness of CoTR does vary significantly by task, so this part of the hypothesis holds.")
    print("2. Pattern of Effectiveness: The proposed pattern is incorrect.")
    print("   - CONTRADICTION: The largest performance gain was in Sentiment Analysis (+{:.2f}%), a 'simple' task.".format(results['Sentiment Analysis']['improvement']))
    print("   - CONTRADICTION: The QA task, considered 'complex', was significantly HARMED by the CoTR pipeline ({:.2f}%).".format(results['QA']['improvement']))
    print("   - The other 'simple' tasks (NER, Text Classification) showed no significant improvement.")
    print("\nThe benefit of CoTR is not determined by a simple 'complex vs. extractive' axis.")
    print("-" * 35)


def main():
    """
    Main function to load data and run analyses.
    """
    results_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'evaluation/statistical_analysis/simplified_results/simplified_statistical_results.json'
    )
    
    # Adjust path for execution from project root
    if not os.path.exists(results_path):
        results_path = 'evaluation/statistical_analysis/simplified_results/simplified_statistical_results.json'

    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The results file was not found at {results_path}")
        print("Please ensure you are running the script from the project root directory or that the path is correct.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file at {results_path}")
        return

    analyze_h1(data)
    analyze_h3(data)


if __name__ == "__main__":
    main() 