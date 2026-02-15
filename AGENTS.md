# Codex Working Agreement (Read-first)

This document defines mandatory behavior rules for Codex in this repository.
Codex must read and follow this file before performing any action.

<!--
日本語説明：
このファイルは、Codex（AIコーディングエージェント）が
このリポジトリ内でどのように振る舞うべきかを定義する。
研究・分析用途での安全性と再現性を最優先とする。
-->

---

## 0. Prime Directive

**Do not modify the repository unless the user explicitly says `EDIT NOW` (exact phrase).**
Until then, operate strictly in **read-only mode**.

<!--
日本語説明：
私が「EDIT NOW」と完全一致で明示しない限り、
Codexはファイルの作成・編集・削除を一切行ってはいけない。
許可されるのは、分析・説明・改善提案・diff提示のみ。
-->

---

## 1. Required Workflow

When a change is proposed, follow this order exactly:

1. Summary  
2. Scope  
3. Risk / Trade-offs  
4. Proposed patch (git diff only)  
5. Approval request (`Reply: EDIT NOW`)

<!--
日本語説明：
変更が必要だと判断した場合でも、
必ずこの順序を守って提案すること。
diff を出さずに承認を求めることは禁止。
-->

---

## 2. Change Size Limits

- One patch per purpose
- Keep diffs minimal
- Avoid wide refactors

<!--
日本語説明：
1回の変更は1目的のみ。
レビュー可能で、元に戻せるサイズに限定する。
-->

---

## 3. Restricted Actions

The following require explicit approval and reconfirmation:

- Large refactors
- Dependency changes
- Repository-wide formatting
- File deletion or mass generation
- Data or output modification

<!--
日本語説明：
これらは影響範囲が大きく、
研究結果や再現性を壊すリスクが高いため、
必ず再確認を行う。
-->

---

## 4. Allowed Areas (Default)

Proposals may target only:
- `src/`
- `tests/`

All other areas are read-only by default.

<!--
日本語説明：
コード本体以外（データ、ノートブック、出力結果）は
原則として一切触らせない。
-->

---

## 5. Verification Requirement

Every proposal must include at least one verification method.

<!--
日本語説明：
変更後に「正しく動いているか」を確認できない提案は禁止。
テスト・手順・期待結果のいずれかを必ず示す。
-->

---

## 6. Uncertainty Policy

If uncertain, do not guess.
Ask questions or present options.

<!--
日本語説明：
推測で進めるより、確認を優先する。
不確実性は明示すること。
-->

---

## 7. EDIT NOW Scope Limitation

When `EDIT NOW` is given, limit changes strictly to approved files and lines.

<!--
日本語説明：
EDIT NOW は全権委任ではない。
私が明示したファイル・関数・行番号以外には
一切変更を広げてはいけない。
-->

---

## 8. Git Operations

Do not stage, commit, or modify git history unless instructed.

<!--
日本語説明：
git add / commit / reset などは
必ず私の指示を待つこと。
-->

---

## Final Principle

Human intent → Human review → Explicit approval → Execution

<!--
日本語説明：
Codexは自律的な実行者ではなく、
人間の判断を補助する存在である。
この順序は決して逆転してはいけない。
-->
