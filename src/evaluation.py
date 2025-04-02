# src/evaluation/evaluate.py

from sklearn.metrics import f1_score

def evaluate_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='weighted')

if __name__ == "__main__":
    true_labels = [0, 1, 2, 2, 1]
    predicted_labels = [0, 2, 2, 2, 1]
    f1 = evaluate_f1(true_labels, predicted_labels)
    print(f"F1 Score: {f1}")