# Stock Analysis & NBA Recommendation Project

本リポジトリは、証券営業における人的資本の異質性と多目的制約を統合した **Next Best Action (NBA) リコメンドモデル** の構築、および株価データの分析を目的とした研究プロジェクトです。

## 1. 研究の背景と目的
- **研究テーマ**: 証券営業における購買確率予測とNBA提案モデルの開発
- **アプローチ**: 営業員の異質性（スキルや特性の違い）や多目的な制約条件を考慮した強化学習アルゴリズムの実装
- **目標**: 組織内でのAIエージェント活用による学習環境の整備と人材育成への貢献

## 2. 主な技術スタック
### 開発環境
- **OS**: Windows 11 / WSL2 (Ubuntu)
- **Hardware**: Alienware M18 R2 (NVIDIA GeForce RTX 4090 Laptop GPU)
- **Conda Environment**: `jupyter_env`

### 使用モデル・アルゴリズム
- **機械学習**: LightGBM, XGBoost, TabNet
- **強化学習**: REINFORCE (Recycle Robot等での実装経験に基づく)
- **ディープラーニング**: Transformers
- **GPU最適化**: RAPIDS (cuDF, cuML), Triton Inference Server

## 3. ディレクトリ構成
- `src/`: データ処理およびモデル学習のソースコード
- `data/`: 証券データ・株価データ（※`.gitignore` により非公開）
- `data/output`: 分析結果・学習済みモデルの出力先
- `.gitignore`: 不要なファイル（CSV、環境設定等）の除外設定
- `main.py`: プロジェクトのメイン実行ファイル

## 4. 環境構築
ターミナルで以下のコマンドを実行し、環境を有効化して使用します。

```bash
# コンダ環境の有効化
conda activate jupyter_env
5. 実績・活動
NVIDIA Student Ambassador (AI/DS Team): 2026年4月本格始動予定

NEC主催データ分析コンペ: 2025年夏季 優秀賞受賞

© 2026 Akifumi Goto (Shiga University / SMBC Nikko Securities Inc.)