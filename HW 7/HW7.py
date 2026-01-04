```python
"""
Homework Originality Label Generator (Template Builder)

What it does:
- Creates a ready-to-paste “originality / source” statement for EACH question.
- Supports:
  1) 全部原創
  2) 全部複製（沒修改）
  3) 除了某些題以外都是原創（那些題會列出參考網址/修改內容）
  4) 每題逐一填寫（原創/參考/修改/複製/用AI並附公開網址）

How to use (quick):
1) Run this file.
2) Enter total number of questions.
3) Choose a mode.
4) Follow prompts.
5) Copy the final output to submit.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QuestionNote:
    q: int
    label: str  # 原創 / 參考 / 修改 / 複製 / 用AI
    source: Optional[str] = None  # 參考誰/網址/複製誰
    change: Optional[str] = None  # 修改了什麼內容
    understood: Optional[str] = None  # 有看懂/沒看懂（通常針對複製）
    ai_url: Optional[str] = None  # 用 AI 的公開網址


def _input_int(prompt: str, min_val: int = 1) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if v < min_val:
                print(f"Please enter an integer >= {min_val}.")
                continue
            return v
        except ValueError:
            print("Please enter a valid integer.")


def _input_choice(prompt: str, choices: List[str]) -> str:
    choices_lower = {c.lower(): c for c in choices}
    while True:
        s = input(prompt).strip().lower()
        if s in choices_lower:
            return choices_lower[s]
        print(f"Choose one of: {', '.join(choices)}")


def build_all_original(n: int) -> str:
    lines = []
    lines.append("全部原創")
    lines.append("")
    lines.append("（提醒：每一題都要標示。以下為逐題標示：）")
    for i in range(1, n + 1):
        lines.append(f"第 {i} 題：原創（自己寫的）")
    return "\n".join(lines)


def build_all_copied(n: int) -> str:
    who = input("全部複製自誰/哪個來源？（可填人名或網址）: ").strip()
    understood = _input_choice("有看懂嗎？輸入: understood / not_understood : ", ["understood", "not_understood"])
    understood_text = "有看懂" if understood == "understood" else "沒看懂"
    lines = []
    lines.append(f"全部複製 {who} 的沒修改（{understood_text}）")
    lines.append("")
    lines.append("（提醒：每一題都要標示。以下為逐題標示：）")
    for i in range(1, n + 1):
        lines.append(f"第 {i} 題：複製（來源：{who}；{understood_text}；未修改）")
    return "\n".join(lines)


def build_mostly_original(n: int) -> str:
    lines = []
    lines.append("除了下列題目以外，都是原創")
    lines.append("")

    k = _input_int("有幾題不是原創（參考/修改/複製/用AI）？: ", min_val=1)
    non_original = {}

    for _ in range(k):
        q = _input_int(f"輸入題號（1~{n}）: ", min_val=1)
        if q > n:
            print(f"題號超過總題數 {n}，請重輸入。")
            continue

        label = _input_choice(
            "此題類型：original / reference / modify / copy / ai : ",
            ["original", "reference", "modify", "copy", "ai"],
        )

        if label == "original":
            non_original[q] = QuestionNote(q=q, label="原創（自己寫的）")
            continue

        if label == "reference":
            src = input("參考誰/什麼網路資源（人名或網址）: ").strip()
            non_original[q] = QuestionNote(q=q, label="參考", source=src)
            continue

        if label == "modify":
            src = input("修改來源（人名或網址）: ").strip()
            change = input("修改了哪些內容（簡述）: ").strip()
            non_original[q] = QuestionNote(q=q, label="修改", source=src, change=change)
            continue

        if label == "copy":
            src = input("直接複製誰的/哪個來源（人名或網址）: ").strip()
            understood = _input_choice("有看懂嗎？輸入: understood / not_understood : ", ["understood", "not_understood"])
            understood_text = "有看懂" if understood == "understood" else "沒看懂"
            changed = _input_choice("有改過嗎？輸入: changed / unchanged : ", ["changed", "unchanged"])
            change_text = "有改過" if changed == "changed" else "沒改過"
            non_original[q] = QuestionNote(
                q=q, label="複製", source=src, understood=understood_text, change=change_text
            )
            continue

        if label == "ai":
            ai_url = input("用 AI 的公開網址（必填，例如分享連結）: ").strip()
            change = input("你有做哪些修改/整理（簡述，可留空）: ").strip()
            non_original[q] = QuestionNote(q=q, label="用AI", ai_url=ai_url, change=change or None)
            continue

    # header: list non-original items
    for q in sorted(non_original.keys()):
        note = non_original[q]
        if note.label.startswith("原創"):
            lines.append(f"第 {q} 題：原創（自己寫的）")
        elif note.label == "參考":
            lines.append(f"第 {q} 題參考 {note.source}（自行撰寫，未剪貼）")
        elif note.label == "修改":
            lines.append(f"第 {q} 題參考 {note.source}，修改了 {note.change}")
        elif note.label == "複製":
            lines.append(f"第 {q} 題複製 {note.source}（{note.understood}；{note.change}）")
        elif note.label == "用AI":
            if note.change:
                lines.append(f"第 {q} 題用 AI（公開網址：{note.ai_url}），並自行修改/整理：{note.change}")
            else:
                lines.append(f"第 {q} 題用 AI（公開網址：{note.ai_url}）")

    lines.append("")
    lines.append("（逐題標示完整版：）")
    for i in range(1, n + 1):
        if i not in non_original:
            lines.append(f"第 {i} 題：原創（自己寫的）")
        else:
            note = non_original[i]
            if note.label.startswith("原創"):
                lines.append(f"第 {i} 題：原創（自己寫的）")
            elif note.label == "參考":
                lines.append(f"第 {i} 題：參考（來源：{note.source}；自行撰寫，未剪貼）")
            elif note.label == "修改":
                lines.append(f"第 {i} 題：修改（來源：{note.source}；修改內容：{note.change}）")
            elif note.label == "複製":
                lines.append(f"第 {i} 題：複製（來源：{note.source}；{note.understood}；{note.change}）")
            elif note.label == "用AI":
                extra = f"；自行修改/整理：{note.change}" if note.change else ""
                lines.append(f"第 {i} 題：用AI（公開網址：{note.ai_url}{extra}）")

    return "\n".join(lines)


def build_per_question(n: int) -> str:
    notes: List[QuestionNote] = []
    for i in range(1, n + 1):
        print(f"\n--- 第 {i} 題 ---")
        label = _input_choice(
            "類型：original / reference / modify / copy / ai : ",
            ["original", "reference", "modify", "copy", "ai"],
        )

        if label == "original":
            notes.append(QuestionNote(q=i, label="原創（自己寫的）"))
        elif label == "reference":
            src = input("參考誰/什麼網路資源（人名或網址）: ").strip()
            notes.append(QuestionNote(q=i, label="參考", source=src))
        elif label == "modify":
            src = input("修改來源（人名或網址）: ").strip()
            change = input("修改了哪些內容（簡述）: ").strip()
            notes.append(QuestionNote(q=i, label="修改", source=src, change=change))
        elif label == "copy":
            src = input("直接複製誰的/哪個來源（人名或網址）: ").strip()
            understood = _input_choice("有看懂嗎？輸入: understood / not_understood : ", ["understood", "not_understood"])
            understood_text = "有看懂" if understood == "understood" else "沒看懂"
            changed = _input_choice("有改過嗎？輸入: changed / unchanged : ", ["changed", "unchanged"])
            change_text = "有改過" if changed == "changed" else "沒改過"
            notes.append(QuestionNote(q=i, label="複製", source=src, understood=understood_text, change=change_text))
        elif label == "ai":
            ai_url = input("用 AI 的公開網址（必填，例如分享連結）: ").strip()
            change = input("你有做哪些修改/整理（簡述，可留空）: ").strip()
            notes.append(QuestionNote(q=i, label="用AI", ai_url=ai_url, change=change or None))

    lines = []
    # auto add summary header if all original / all copied etc.
    all_original = all(n.label.startswith("原創") for n in notes)
    all_copy_unchanged = all(n.label == "複製" and n.change == "沒改過" for n in notes)

    if all_original:
        lines.append("全部原創")
        lines.append("")
    elif all_copy_unchanged:
        # if all copied, pick first source as headline
        first_src = notes[0].source or "（未填來源）"
        lines.append(f"全部複製 {first_src} 的沒修改")
        lines.append("")

    for nte in notes:
        if nte.label.startswith("原創"):
            lines.append(f"第 {nte.q} 題：原創（自己寫的）")
        elif nte.label == "參考":
            lines.append(f"第 {nte.q} 題：參考（來源：{nte.source}；自行撰寫，未剪貼）")
        elif nte.label == "修改":
            lines.append(f"第 {nte.q} 題：修改（來源：{nte.source}；修改內容：{nte.change}）")
        elif nte.label == "複製":
            lines.append(f"第 {nte.q} 題：複製（來源：{nte.source}；{nte.understood}；{nte.change}）")
        elif nte.label == "用AI":
            extra = f"；自行修改/整理：{nte.change}" if nte.change else ""
            lines.append(f"第 {nte.q} 題：用AI（公開網址：{nte.ai_url}{extra}）")

    return "\n".join(lines)


def main():
    print("=== 作業『每題標示原創/參考/修改/複製/AI』文字產生器 ===\n")
    n = _input_int("你的作業共有幾題？: ", min_val=1)

    print("\n選擇模式：")
    print("1) 全部原創")
    print("2) 全部複製（沒修改）")
    print("3) 除了部分題目以外都是原創（列出那些題）")
    print("4) 每題逐一填寫（最完整）")

    mode = _input_choice("輸入 1 / 2 / 3 / 4 : ", ["1", "2", "3", "4"])

    if mode == "1":
        out = build_all_original(n)
    elif mode == "2":
        out = build_all_copied(n)
    elif mode == "3":
        out = build_mostly_original(n)
    else:
        out = build_per_question(n)

    print("\n\n========== 產生結果（直接複製貼上） ==========\n")
    print(out)
    print("\n===========================================\n")


if __name__ == "__main__":
    main()
```
