import os
import streamlit as st
from typing import Literal

# ---------------------------
# アプリ設定
# ---------------------------
st.title("サンプルアプリ③: LLM専門家モード切替デモ")

# 概要と操作方法
st.write("##### 概要")
st.write("""
- 入力フォームにテキストを入力し、ラジオボタンで **LLMの専門家ロール（A/B）** を選択します。  
- 選択ロールに応じて **システムメッセージ（役割指示）を切り替え**、LangChain 経由で LLM にプロンプトを渡して回答を表示します。  
- 実装は **ChatPromptTemplate → Chat LLM → StrOutputParser** の最小チェーン（Lesson8相当）です。
""")

st.write("##### 操作方法")
st.write("""
1. **専門家の種類（A/B）** を選択  
2. 下の **入力フォーム** に質問/依頼内容を入力  
3. **「実行」** ボタンで回答を表示
""")

st.info("※ OpenAI APIキー（環境変数 `OPENAI_API_KEY`）が必要です。モデルは `gpt-4o-mini` を既定使用します。")

# ---------------------------
# ラジオボタン（専門家選択）
# ---------------------------
role_label = st.radio(
    "LLMに振る舞わせる専門家の種類を選択してください。",
    options=["A: IT導入/業務改革のプロジェクトマネージャー", "B: データアナリスト / BIコンサルタント"],
    index=0
)
# "A" or "B" のキー化
ExpertKey = Literal["A", "B"]
role_key: ExpertKey = "A" if role_label.startswith("A") else "B"

st.divider()

# ---------------------------
# 入力フォーム（※1つだけ）
# ---------------------------
input_text = st.text_area(
    label="LLMに渡す入力テキストを入力してください。",
    placeholder="例）出張精算フローの承認リードタイムを短縮したい。現状課題の整理と実行計画を提示してほしい。"
)

# ---------------------------
# LangChain 用の設定
# ---------------------------
# A/B のシステムメッセージ定義（必要なら編集可）
EXPERT_SYSTEM_MESSAGES = {
    "A": (
        "あなたはIT導入・業務改革のプロジェクトマネージャー（PM）の第一人者です。"
        "ユーザーの依頼に対し、(1)目的、(2)要件整理、(3)現状課題と制約、"
        "(4)実行計画（優先順位/体制/スケジュール/リスク）、(5)期待効果（定量/定性）"
        "の流れで、日本語で具体的かつ実務で再現可能な提案を行ってください。"
        "過剰な仮定は避け、必要最小限の前提を明示してください。"
    ),
    "B": (
        "あなたはデータアナリスト/BIコンサルタントの専門家です。"
        "ユーザーの依頼に対し、(1)ビジネスKPI定義、(2)指標・可視化設計、"
        "(3)データ収集/前処理、(4)分析手法（仮説/検定/モデル）、(5)示唆と次アクション"
        "の順で、日本語で手順を具体的に提示してください。"
        "効果測定（A/Bテスト、ベースライン比較）にも簡潔に触れてください。"
    ),
}

def _init_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    OpenAI Chat モデルを初期化して返す。
    - OPENAI_API_KEY は環境変数から取得（未設定なら例外）
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")
    return ChatOpenAI(model=model_name, temperature=temperature)

# ---------------------------
# 要件の関数：
#  「入力テキスト」と「ラジオボタン選択値」を引数に取り、
#   LLMからの回答（文字列）を戻り値として返す
# ---------------------------
def ask_llm_by_role(user_input: str, role: ExpertKey) -> str:
    if role not in EXPERT_SYSTEM_MESSAGES:
        raise ValueError("role には 'A' または 'B' を指定してください。")
    if not user_input or not user_input.strip():
        return "入力テキストが空です。内容を入力してください。"

    system_msg = EXPERT_SYSTEM_MESSAGES[role]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{user_input}")
        ]
    )
    chain = prompt | _init_llm() | StrOutputParser()
    return chain.invoke({"user_input": user_input})

# ---------------------------
# 実行ボタン
# ---------------------------
if st.button("実行"):
    st.divider()
    if not input_text.strip():
        st.error("入力テキストを入力してから「実行」ボタンを押してください。")
    else:
        try:
            with st.spinner("LLMに問い合わせ中..."):
                answer = ask_llm_by_role(input_text, role_key)
            st.success("回答を取得しました。")
            st.write("##### 回答")
            st.write(answer)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")