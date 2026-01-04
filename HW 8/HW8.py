```python
"""
README.md Generator for Homework (Per-Question Completion Method)

Meets requirements:
- MUST explain how each question was completed.
- For each question: choose one of
  原創 / 參考某人 / 參考網路(網址) / 使用AI(例如 ChatGPT, Gemini)
- Hard questions get MORE detailed explanation automatically (you decide difficulty 1-5).
- Outputs a ready-to-submit README.md file.

Run:
python readme_generator.py
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass
class QuestionInfo:
    number: int
    title: str
    category: str  # 原創 / 參考某人 / 參考網路 / 使用AI
    source: Optional[str]  # person name or URL or AI tool name
    difficulty: int  # 1-5
    method_short: str  # always required
    method_detail: Optional[str] = None  # required if difficulty is high
    ai_public_url: Optional[str] = None  # required if using AI


def _ask_int(prompt: str, min_v: int, max_v: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if min_v <= v <= max_v:
                return v
            print(f"Please enter a number between {min_v} and {max_v}.")
        except ValueError:
            print("Please enter a valid integer.")


def _ask_nonempty(prompt: str) -> str:
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("This field cannot be empty.")


def _ask_choice(prompt: str, choices: List[str]) -> str:
    normalized = {c.lower(): c for c in choices}
    while True:
        s = input(prompt).strip().lower()
        if s in normalized:
            return normalized[s]
        print(f"Choose one of: {', '.join(choices)}")


def _category_to_label(cat: str) -> str:
    # internal mapping if needed; keep Traditional Chinese output
    return cat


def collect_questions() -> List[QuestionInfo]:
    print("=== README.md 產生器：逐題說明完成方法（含原創/參考/AI）===\n")

    total = _ask_int("作業總共有幾題？(>=1): ", 1, 10_000)
    questions: List[QuestionInfo] = []

    print("\n說明：每一題都要填『完成方法』。難度 4~5 視為較難，會要求更詳細說明。\n")
    for i in range(1, total + 1):
        print(f"\n--- 第 {i} 題 ---")
        title = input("題目簡短描述（可留空，建議填，例如：'Stack 구현'）: ").strip() or f"Question {i}"

        category = _ask_choice(
            "類型輸入：original / person / web / ai : ",
            ["original", "person", "web", "ai"],
        )

        if category == "original":
            cat_label = "原創"
            source = None
        elif category == "person":
            cat_label = "參考某人"
            source = _ask_nonempty("參考誰（姓名/同學/老師/作者等）: ")
        elif category == "web":
            cat_label = "參考網路"
            source = _ask_nonempty("參考網址（URL）: ")
        else:
            cat_label = "使用AI"
            source = _ask_nonempty("使用哪個 AI（例如 ChatGPT / Gemini / Claude）: ")

        difficulty = _ask_int("難度 (1~5)：", 1, 5)

        method_short = _ask_nonempty(
            "完成方法（必填，簡短 2~5 句）：\n"
            "例如：先讀題→列出已知與未知→選擇演算法/公式→實作/計算→驗算。\n> "
        )

        method_detail = None
        if difficulty >= 4:
            method_detail = _ask_nonempty(
                "較難題目需要更詳細說明（至少 5~10 句，可含步驟/推導/測試方式/遇到的錯誤與修正）：\n> "
            )

        ai_public_url = None
        if cat_label == "使用AI":
            ai_public_url = _ask_nonempty("AI 公開分享網址（必填）: ")

        questions.append(
            QuestionInfo(
                number=i,
                title=title,
                category=cat_label,
                source=source,
                difficulty=difficulty,
                method_short=method_short,
                method_detail=method_detail,
                ai_public_url=ai_public_url,
            )
        )

    return questions


def build_readme_md(
    course: str,
    assignment_name: str,
    author_name: str,
    questions: List[QuestionInfo],
) -> str:
    today = date.today().isoformat()

    lines: List[str] = []
    lines.append(f"# {assignment_name}")
    lines.append("")
    lines.append("## 基本資訊")
    lines.append(f"- 課程：{course}")
    lines.append(f"- 作者：{author_name}")
    lines.append(f"- 日期：{today}")
    lines.append("")
    lines.append("## 完成方式與來源註記（每一題）")
    lines.append("")
    lines.append("> 規則：每一題都標示 **原創 / 參考某人 / 參考網路(網址) / 使用AI**，並說明完成方法。")
    lines.append("> 若為較難題目（難度 4~5），提供更詳細的說明。")
    lines.append("")

    for q in questions:
        lines.append(f"### 第 {q.number} 題：{q.title}")
        lines.append("")
        lines.append(f"- 類型：**{_category_to_label(q.category)}**")

        if q.category == "參考某人":
            lines.append(f"- 參考對象：{q.source}")
        elif q.category == "參考網路":
            lines.append(f"- 參考網址：{q.source}")
        elif q.category == "使用AI":
            lines.append(f"- 使用 AI：{q.source}")
            lines.append(f"- AI 公開網址：{q.ai_public_url}")

        lines.append(f"- 難度：{q.difficulty}/5")
        lines.append("")
        lines.append("**完成方法（簡述）**")
        lines.append("")
        lines.append(q.method_short.strip())
        lines.append("")

        if q.difficulty >= 4 and q.method_detail:
            lines.append("**較難題目：更詳細說明**")
            lines.append("")
            lines.append(q.method_detail.strip())
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=== README.md 產生器（逐題完成方法 + 原創/參考/AI）===\n")

    course = input("課程名稱（可留空）: ").strip() or "（未填）"
    assignment_name = input("作業名稱（可留空）: ").strip() or "Homework"
    author_name = input("作者姓名（可留空）: ").strip() or "（未填）"

    questions = collect_questions()

    readme = build_readme_md(
        course=course,
        assignment_name=assignment_name,
        author_name=author_name,
        questions=questions,
    )

    out_path = "README.md"
    write_file(out_path, readme)

    print("\n✅ 已產生 README.md（同資料夾內）")
    print("你也可以直接複製以下內容貼到 README.md：\n")
    print("========== README.md BEGIN ==========\n")
    print(readme)
    print("=========== README.md END ===========\n")


if __name__ == "__main__":
    main()
```
