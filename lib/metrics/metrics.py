from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred, labels):
    """
    Dict containing the metrics to evaluate trained model
    on. Template and usage obtained from: https://huggingface.co/docs/transformers/training
    """
    labels = labels
    preds = pred.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_score_micro = f1_score(labels, preds, average='micro')
    f1_score_macro = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1_score_micro': f1_score_micro,
        'f1_score_macro': f1_score_macro
    }