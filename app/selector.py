from __future__ import annotations

import random
from typing import Dict

import pandas as pd


def select_random_card(pool: pd.DataFrame) -> pd.Series | None:
    """Select one random card row from pool or None if empty."""
    if pool.empty:
        return None
    return pool.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]


def select_weighted_card(
    cards: pd.DataFrame,
    progress: pd.DataFrame,
    weights: Dict[int, int],
) -> pd.Series | None:
    """Select one card using score_class weights (0/1/2)."""
    merged = cards.merge(progress, on="id", how="left")
    merged["score_class"] = merged["score_class"].fillna(0).astype(int)

    pools = {
        0: merged[merged["score_class"] == 0],
        1: merged[merged["score_class"] == 1],
        2: merged[merged["score_class"] == 2],
    }

    weighted_classes = [
        score_class
        for score_class, pool in pools.items()
        if not pool.empty and weights.get(score_class, 0) > 0
    ]
    if not weighted_classes:
        return None

    class_weights = [weights.get(score_class, 0) for score_class in weighted_classes]
    chosen_class = random.choices(weighted_classes, weights=class_weights, k=1)[0]
    return select_random_card(pools[chosen_class])
