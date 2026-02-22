
# app.py
# ------------------------------------------------------------
# シンプルな「専門家切り替え」UI + LangChain チェーンのサンプル
#  - 入力テキストとラジオボタンの選択値を引数に取り、LLM応答を返す関数を定義
#  - ラジオボタン選択に応じてシステムメッセージ（専門家の振る舞い）を切替
#  - 画面にアプリ概要・操作説明を明示表示
# ------------------------------------------------------------

import streamlit as st

EXPERT_SYSTEM_MESSAGES = {
    # A: 「IT導入・業務改革のプロジェクトマネージャー（PM）」
    "A": (
        "あなたはIT導入・業務改革のプロジェクトマネージャー（PM）の第一人者です。"
        "ユーザーの問いに対し、(1)目的の明確化、(2)要件整理、(3)現状課題と制約、"
        "(4)実行計画（優先順位/スケジュール/体制/リスク）、(5)期待効果（定量/定性）"
        "の順で、簡潔かつ実務で再現可能な提案を日本語で行ってください。"
        "不明点は過剰に仮定せず、必要最小限の前提を置いて補足してください。"
    ),
    # B: 「データアナリスト / BIコンサルタント」
    "B": (
        "あなたはデータアナリスト/BIコンサルタントの専門家です。"
        "ユーザーの問いに対し、(1)ビジネスKPIの定義、(2)指標設計と可視化方針、"
        "(3)データ収集/前処理、(4)分析手法（仮説/検定/モデル）、(5)示唆と次アクション"
        "の流れで、手順を具体的に日本語で示してください。"
        "効果測定の観点（A/Bテストやベースライン比較）も簡潔に触れてください。"
    ),
}


# ========= 重要：要件の関数 =========
def ask_llm_by_role(input_text: str, role_key: ExpertKey) -> str:
    """
    要件:
      - 「入力テキスト」と「ラジオボタンの選択値（A/B）」を引数に取り、
        LLMの回答（文字列）を戻り値として返す関数。

    :param input_text: 画面の入力フォームからのテキスト
    :param role_key:   ラジオボタンで選んだ専門家キー（"A" または "B"）
    :return: LLMからの回答テキスト
    """
    if role_key not in EXPERT_SYSTEM_MESSAGES:
        raise ValueError("不正な専門家キーです。'A' または 'B' を指定してください。")
    if not input_text or not input_text.strip():
        return "入力テキストが空のようです。内容を記入してください。"

    # --- LLM とチェーンの用意（Lesson8相当の最小構成） ---
    llm = _init_llm()
    system_msg = EXPERT_SYSTEM_MESSAGES[role_key]

    # ChatPromptTemplate: system（役割） + human（ユーザー入力）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{user_input}"),
        ]
    )

    # チェーン = プロンプト → LLM → 文字列パーサ
    chain = prompt | llm | StrOutputParser()

    # 実行
    result_text: str = chain.invoke({"user_input": input_text})
    return result_text


# ========= Streamlit UI =========
st.set_page_config(page_title="専門家切替 LLM デモ", page_icon="🧠")
st.title("🧠 専門家切替 LLM デモ（LangChain / Lesson8相当）")

with st.expander("ℹ️ このアプリについて / 操作方法", expanded=True):
    st.markdown(
        """
**概要**  
- テキスト入力とラジオボタンで専門家の種類（A/B）を選び、LLMに回答させるデモです。  
- 選択した専門家に応じて **システムメッセージ（役割）** を切り替え、**回答の観点** を変えます。  
- LangChain の **ChatPromptTemplate → Chat LLM → StrOutputParser** の最小構成（Lesson8相当）で実装しています。

**使い方**  
1. 右のラジオボタンで、LLMの専門家の種類（A/B）を選びます  
   - **A**: IT導入/業務改革の**プロジェクトマネージャー（PM）**  
   - **B**: **データアナリスト/BIコンサルタント**  
2. 下のテキストエリアに質問や依頼内容を入力します  
3. **「送信して回答を取得」** ボタンを押すと、結果が下部に表示されます

**注意**  
- 実行には OpenAI API キー（`OPENAI_API_KEY`）が必要です。`.env` または環境変数で設定してください。  
- モデル名は既定で `gpt-4o-mini` を使用しています。環境に合わせて変更してください。
        """
    )

# --- サイドバー（専門家選択） ---
st.sidebar.header("専門家の選択")
role_label = st.sidebar.radio(
    "LLMに振る舞わせる専門家",
    options=["A: IT導入/業務改革PM", "B: データアナリスト/BIコンサルタント"],
    index=0,
    help="選択に応じて、LLMに渡すシステムメッセージ（役割）を切り替えます。",
)

# ラベルからキー（"A" / "B"）を取り出す簡易パース
role_key: ExpertKey = "A" if role_label.startswith("A") else "B"

# --- 本文（入力エリア） ---
st.subheader("入力テキスト")
user_text = st.text_area(
    "質問や依頼内容を入力してください（例：社内ポータルの改善計画を立てたい、KPIダッシュボードを設計したい等）",
    height=180,
    placeholder="例）出張精算の承認リードタイムを30%短縮したい。現状の課題整理と実行計画（優先順位、役割分担、リスク）をPMの観点で提案してください。",
)

# --- 送信ボタン ---
if st.button("送信して回答を取得", type="primary"):
    try:
        with st.spinner("LLMに問い合わせ中…"):
            answer = ask_llm_by_role(user_text, role_key)
        st.success("回答を取得しました")
        st.markdown("### 回答")
        st.write(answer)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# --- フッタ補足 ---
st.markdown(
    """
---
**実装メモ**  
- `ask_llm_by_role(input_text, role_key)` が、要件の**関数**（入力テキスト & ラジオ選択 → LLM応答を返す）。  
- 専門家の切替は `EXPERT_SYSTEM_MESSAGES` で定義。A/B の役割文を編集するだけで拡張可能。  
- LangChain の**最小チェーン**: `ChatPromptTemplate` → `ChatOpenAI` → `StrOutputParser`。  
- 例外時は `st.error` で通知。API Key 未設定時は起動後にエラー表示。
"""
)
