import pandas as pd
import numpy as np

def mrr(ranked_methods, ground_truth_methods):
    """
    Returns the reciprocal of the rank of the first correctly identified ground truth
    """
    for rank, method in enumerate(ranked_methods, start=1):
        if method in ground_truth_methods:
            return 1.0 / rank
    return 0.0

def top_k_acc(ranked_methods, ground_truth_methods, k):
    """
    Verify that the top K prediction results include at least one ground truth
    """
    top_k_methods = ranked_methods[:k]
    if set(top_k_methods) & set(ground_truth_methods):
        return 1.0
    return 0.0

def evaluate_project_results(df_results, model_columns):
    """
    param df_results: A DataFrame containing the project, bug ID, method name, predicted scores for each model, 
                      and whether the data is ground truth
    param model_columns: i.e. ['Score_BM25', 'Score_Vanilla_GNN', 'Score_Graphectory']
    """
    metrics_summary = {model: {'MRR': [], 'Top-1': [], 'Top-3': [], 'Top-5': []} for model in model_columns}
    
    # Capturer First Failure
    grouped = df_results.groupby('Bug_ID')
    
    for bug_id, group in grouped:
        # List of methods where actual bugs occurred (Ground Truth)
        ground_truth = group[group['Is_Ground_Truth'] == 1]['Method_Name'].tolist()
        
        # Abnormal data
        if not ground_truth:
            continue
            
        for model in model_columns:
            # The score for this model
            ranked_list = group.sort_values(by=model, ascending=False)['Method_Name'].tolist()
            
            metrics_summary[model]['MRR'].append(mrr(ranked_list, ground_truth))
            metrics_summary[model]['Top-1'].append(top_k_acc(ranked_list, ground_truth, k=1))
            metrics_summary[model]['Top-3'].append(top_k_acc(ranked_list, ground_truth, k=3))
            metrics_summary[model]['Top-5'].append(top_k_acc(ranked_list, ground_truth, k=5))
            
    # Calculating the Mean by Model
    final_result = []
    for model, metrics in metrics_summary.items():
        final_result.append({
            'Model': model.replace('Score_', ''),
            'MRR': np.mean(metrics['MRR']),
            'Top-1 Acc': np.mean(metrics['Top-1']),
            'Top-3 Acc': np.mean(metrics['Top-3']),
            'Top-5 Acc': np.mean(metrics['Top-5'])
        })
        
    return pd.DataFrame(final_report)

if __name__ == "__main__":
    
    # Extract data from csv
    """
    mock_data = {
        'Project': [..., ..., ..., ...],
        'Bug_ID': [..., ..., ..., ...],
        'Method_Name': [..., ..., ..., ...],
        'Score_BM25': [..., ..., ..., ...], # Information Retrieval
        'Score_Vanilla_GNN': [..., ..., ..., ...], # Structural GNN
        'Score_Graphectory': [..., ..., ..., ...], # Trajectory-based GNN
        'Is_Ground_Truth': [..., ..., ..., ...] # method_C is the method that actually triggers the first failure
    }
    
    df = pd.DataFrame(mock_data)
    """

    # List of models to be evaluated
    models_to_evaluate = ['Score_BM25', 'Score_Vanilla_GNN', 'Score_Graphectory']
    evaluation_results = evaluate_project_results(df, models_to_evaluate)
    
    pd.options.display.float_format = '{:.4f}'.format
    print("Final Evaluation Results")
    print(evaluation_results.to_string(index=False))