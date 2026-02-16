from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import calculate_similarity, classify_score
from selector import select_random_card, select_weighted_card

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROGRESS_PATH = DATA_DIR / "progress.csv"


def load_cards() -> pd.DataFrame:
    csv_paths = sorted(DATA_DIR.glob("flashcards_*.csv"))
    if not csv_paths:
        return pd.DataFrame(columns=["id", "question", "answer"])

    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path, header=None, names=["question", "answer"])
        frame = frame.reset_index().rename(columns={"index": "source_row"})
        frame["source_path"] = str(path)
        frames.append(frame)
    cards = pd.concat(frames, ignore_index=True)
    cards.index.name = "id"
    cards = cards.reset_index()
    return cards


def load_progress(card_count: int) -> pd.DataFrame:
    if PROGRESS_PATH.exists():
        progress = pd.read_csv(PROGRESS_PATH)
        progress = progress[["id", "score_class"]]
    else:
        progress = pd.DataFrame(columns=["id", "score_class"])

    existing_ids = set(progress["id"].tolist()) if not progress.empty else set()
    missing_ids = [card_id for card_id in range(card_count) if card_id not in existing_ids]
    if missing_ids:
        additions = pd.DataFrame({"id": missing_ids, "score_class": [0] * len(missing_ids)})
        progress = pd.concat([progress, additions], ignore_index=True)

    progress["score_class"] = progress["score_class"].fillna(0).astype(int)
    progress = progress.sort_values("id").reset_index(drop=True)
    progress.to_csv(PROGRESS_PATH, index=False)
    return progress


def save_progress(progress: pd.DataFrame) -> None:
    progress.to_csv(PROGRESS_PATH, index=False)


def update_model_answer(cards: pd.DataFrame, card_id: int, new_answer: str) -> bool:
    matches = cards[cards["id"] == card_id]
    if matches.empty:
        return False

    card = matches.iloc[0]
    source_path = Path(card["source_path"])
    source_row = int(card["source_row"])

    source_df = pd.read_csv(source_path, header=None, names=["question", "answer"])
    if source_row >= len(source_df):
        return False

    source_df.at[source_row, "answer"] = new_answer
    source_df.to_csv(source_path, index=False, header=False)
    cards.loc[cards["id"] == card_id, "answer"] = new_answer
    return True


def ensure_current_question(
    cards: pd.DataFrame,
    progress: pd.DataFrame,
    weights: dict[int, int],
    weights_valid: bool,
) -> pd.Series | None:
    current_id = st.session_state.get("current_id")
    if current_id is not None:
        matches = cards[cards["id"] == current_id]
        if not matches.empty:
            return matches.iloc[0]
    if weights_valid:
        return select_weighted_card(cards, progress, weights)
    return select_random_card(cards)


st.set_page_config(page_title="損保2次試験用フラッシュカード")

st.title("損保2次試験 フラッシュカード")

cards = load_cards()
if cards.empty:
    st.warning("問題データが見つかりません。data/flashcards_*.csv を確認してください。")
    st.stop()

progress = load_progress(len(cards))

status = cards.merge(progress, on="id", how="left")
status["score_class"] = status["score_class"].fillna(0).astype(int)
total_cards = len(status)
count0 = int((status["score_class"] == 0).sum())
count1 = int((status["score_class"] == 1).sum())
count2 = int((status["score_class"] == 2).sum())
if total_cards > 0:
    pct0 = round(count0 / total_cards * 100)
    pct1 = round(count1 / total_cards * 100)
    pct2 = round(count2 / total_cards * 100)
else:
    pct0 = pct1 = pct2 = 0

st.subheader("ステータス集計")
col_s0, col_s1, col_s2 = st.columns(3)
with col_s0:
    st.metric("未正解(0)", f"{count0}問", f"{pct0}%")
with col_s1:
    st.metric("部分(1)", f"{count1}問", f"{pct1}%")
with col_s2:
    st.metric("正解(2)", f"{count2}問", f"{pct2}%")

st.subheader("出題割合")
col_w0, col_w1, col_w2 = st.columns(3)

with col_w0:
    weight0 = st.number_input(
        "未正解(0)%",
        min_value=0,
        max_value=100,
        value=st.session_state.get("weight0", 75),
        step=5,
    )

with col_w1:
    weight1 = st.number_input(
        "部分(1)%",
        min_value=0,
        max_value=100,
        value=st.session_state.get("weight1", 20),
        step=5,
    )

with col_w2:
    weight2 = st.number_input(
        "正解(2)%",
        min_value=0,
        max_value=100,
        value=st.session_state.get("weight2", 5),
        step=5,
    )

st.session_state["weight0"] = weight0
st.session_state["weight1"] = weight1
st.session_state["weight2"] = weight2

weights_total = weight0 + weight1 + weight2
weights_valid = weights_total == 100
if not weights_valid:
    st.warning("出題割合の合計が100%になるよう設定してください。")

weights = {0: weight0, 1: weight1, 2: weight2}

current = ensure_current_question(cards, progress, weights, weights_valid)
if current is None:
    st.warning("問題を選択できませんでした。")
    st.stop()

st.session_state["current_id"] = int(current["id"])

st.subheader("問題")
st.write(current["question"])

if st.session_state.get("reset_answer"):
    st.session_state["user_answer"] = ""
    st.session_state["reset_answer"] = False

user_answer = st.text_area("あなたの回答", key="user_answer")

col_score, col_next = st.columns(2)

with col_score:
    if st.button("採点"):
        similarity = calculate_similarity(user_answer, str(current["answer"]))
        score_class = classify_score(similarity)
        st.session_state["last_similarity"] = similarity
        st.session_state["last_answer"] = str(current["answer"])
        st.session_state["scored"] = True

        progress.loc[progress["id"] == int(current["id"]), "score_class"] = score_class
        save_progress(progress)

with col_next:
    if st.button("次の問題"):
        if not weights_valid:
            st.warning("出題割合の合計が100%になるよう設定してください。")
        else:
            next_card = select_weighted_card(cards, progress, weights)
            if next_card is None:
                st.warning("選択した出題割合で問題を選べませんでした。")
            else:
                st.session_state["current_id"] = int(next_card["id"])
                st.session_state["reset_answer"] = True
                st.session_state["scored"] = False
                st.rerun()

if st.session_state.get("scored"):
    st.markdown("---")
    st.write(f"一致率: {st.session_state.get('last_similarity', 0)}%")
    edited_answer = st.text_area(
        "模範解答（編集可）",
        value=st.session_state.get("last_answer", ""),
        key="edited_answer",
    )
    col_save, col_mark = st.columns(2)
    with col_save:
        if st.button("模範解答を保存"):
            saved = update_model_answer(cards, int(current["id"]), edited_answer)
            if saved:
                st.session_state["last_answer"] = edited_answer
                st.success("模範解答を保存しました。")
            else:
                st.error("模範解答の保存に失敗しました。")

    with col_mark:
        if st.button("正答として登録(100%)"):
            progress.loc[progress["id"] == int(current["id"]), "score_class"] = 2
            save_progress(progress)
            st.session_state["last_similarity"] = 100
            st.success("100%正解として記録しました。")
