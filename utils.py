import rbo
import numpy as np

def compute_diversity(recommendations: dict) -> float:
    """
    Вычисляет diversity между множествами рекомендаций от разных моделей.

    :param recommendations: словарь вида {model_name: [item_id1, item_id2, ...]}
    :return: доля уникальных айтемов от общего количества (от 0 до 1)
    """
    all_items = []
    for recs in recommendations.values():
        all_items.extend(recs)
    
    if not all_items:
        return 0.0

    unique_items = set(all_items)
    return len(unique_items) / len(all_items)

def add_model_scores(df, heuristic_model, mf_model, nn_model):
    df = df.copy()
    df['heuristic_score'] = df.apply(lambda row: heuristic_model.predict(row['user_id'], row['item_id']), axis=1)
    df['mf_score'] = df.apply(lambda row: mf_model.predict(row['user_id'], row['item_id']), axis=1)
    df['nn_score'] = df.apply(lambda row: nn_model.predict(row['user_id'], row['item_id']), axis=1)
    return df