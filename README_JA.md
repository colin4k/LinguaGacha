<div align=center><img src="https://github.com/user-attachments/assets/de19ec3f-246c-432d-9636-ff16f82b094e" width="256px;"></div>
<div align=center><img src="https://img.shields.io/github/v/release/neavo/LinguaGacha"/>   <img src="https://img.shields.io/github/license/neavo/LinguaGacha"/>   <img src="https://img.shields.io/github/stars/neavo/LinguaGacha"/></div>
<p align='center'>AIの能力を活用して小説、ゲーム、字幕などのテキストをワンクリックで翻訳する次世代のテキスト翻訳ツール</p>

&ensp;
&ensp;

## README 🌍
- [ [中文](./README.md) ] | [ [English](/README_EN.md) ] | [ [日本語](/README_JA.md) ]

## 概要 📢
- [LinguaGacha](https://github.com/neavo/LinguaGacha) (/ˈlɪŋɡwə ˈɡɑːtʃə/)、AIを活用した次世代のテキスト翻訳ツールです
- 箱から出してすぐに使え、（ほぼ）設定不要。機能の強力さは、煩雑な設定を必要としません。
- `中国語`、`英語`、`日本語`、`韓国語`、`ロシア語`、`ドイツ語`、`フランス語`、`イタリア語`など13言語にワンタッチ双方向翻訳対応。
- `字幕`、`電子書籍`、`ゲームテキストなど`、色々なテキストタイプと形式に対応。
- `Claude`、`ChatGPT`、`DeepSeek`、`SakuraLLM` などのローカルおよびオンラインインターフェースをサポート

> <img src="https://github.com/user-attachments/assets/99f7d74e-ab5b-4645-b736-6f665782b4af" style="width: 80%;">

> <img src="https://github.com/user-attachments/assets/c0d7e898-f6fa-432f-a3cd-e231b657c4b5" style="width: 80%;">

## 特別なお知らせ ⚠️
- 翻訳中に [LinguaGacha](https://github.com/neavo/LinguaGacha) を使用する場合は、作品の情報やリリースページの目立つ場所に明確な帰属を含めてください！
- 商業活動や利益を伴うプロジェクトの場合は、[LinguaGacha](https://github.com/neavo/LinguaGacha) を使用する前に、著者に連絡して許可を得てください！

## 主な機能 📌
- 超高速の翻訳速度：字幕は10秒、小説は1分、ゲームは5分で翻訳
- 自動用語集生成により、作品全体で一貫した用語（例：キャラクター名）を保証 `👈👈 独自機能`
- フラッグシップモデル（例：DeepSeek-R1）からローカル小型モデル（例：Qwen2.5-7B）まで、最適な翻訳品質を提供
- テキストのフォーマットと埋め込みコードを `100%` 正確に保持し、後処理作業を大幅に削減 - 埋め込みローカリゼーションに最適 `👈👈 独自機能`

## システム要件 🖥️
- `OpenAI`、`Google`、`Anthropic`、`SakuraLLM` 標準に準拠したAIモデルインターフェースに対応
- [KeywordGacha](https://github.com/neavo/KeywordGacha) と互換性あり `👈👈 AIを活用して用語集をワンクリックで生成する次世代ツール`

## ワークフロー 🛸
- [リリースページ](https://github.com/neavo/LinguaGacha/releases) からアプリケーションをダウンロード
- 信頼できるAIモデルインターフェースを取得（以下のいずれかを選択）：
  - [ローカルAPI - チュートリアル](https://github.com/neavo/OneClickLLAMA)（無料、8GB以上のVRAM GPUが必要、Nvidia推奨）
  - [DeepSeek - チュートリアル](https://github.com/neavo/LinguaGacha/wiki/DeepSeek)（有料、コストパフォーマンスが高く、高速で高品質、GPU不要） `👈👈 推奨`
- ソーステキストを準備：
  - `字幕`/`電子書籍`は通常、前処理が不要
  - `ゲームテキスト`は特定のゲームエンジンに適したツールを使用して抽出が必要
- `app.exe` を実行してアプリケーションを起動：
  - `プロジェクト設定` で必要な設定（ソース/ターゲット言語）を行う
  - 入力フォルダ（デフォルト：`input`）にファイルをコピーし、`翻訳開始` で翻訳を開始

## 使い方チュートリアル - English 📝
- Text Tutorial
  - [Basic Tutorial](https://github.com/neavo/LinguaGacha/wiki/BasicTutorial)　`👈👈 Step-by-step tutorial, easy to follow, a must-read for beginners`
- Video Tutorial
  - [How to Translate RPGMV with LinguaGacha and Translator++ (English)](https://www.youtube.com/watch?v=wtV_IODzi8I)
- Advance Tutorial
  - [Glossary](https://github.com/neavo/LinguaGacha/wiki/Glossary-%E2%80%90-EN)　　[Replacement](https://github.com/neavo/LinguaGacha/wiki/Replacement-%E2%80%90-EN)　　[Incremental Translation](https://github.com/neavo/LinguaGacha/wiki/IncrementalTranslation-%E2%80%90-EN)
  - [ReTranslation](https://github.com/neavo/LinguaGacha/wiki/ReTranslation-%E2%80%90-EN)　　[Expert Config](https://github.com/neavo/LinguaGacha/wiki/ExpertConfig-%E2%80%90-EN)　　[Name Injection](https://github.com/neavo/LinguaGacha/wiki/NameInjection-%E2%80%90-EN)
  - [MTool Optimizer](https://github.com/neavo/LinguaGacha/wiki/MToolOptimizer-%E2%80%90-EN)
  - [Best Practices for High-Quality Translation of RPGMaker Series Engine Games](https://github.com/neavo/LinguaGacha/wiki/BestPracticeForRPGMaker-%E2%80%90-EN)
- You can find more details on each feature in the [Wiki](https://github.com/neavo/LinguaGacha/wiki), and you are welcome to share your experience in the [Discussions](https://github.com/neavo/LinguaGacha/discussions)

## 対応フォーマット 🏷️
- 入力フォルダ内のすべての対応ファイル（サブディレクトリを含む）を処理：
  - 字幕 (.srt .ass)
  - 電子書籍 (.txt .epub)
  - Markdown（.md）
  - [RenPy](https://www.renpy.org) エクスポート (.rpy)
  - [MTool](https://mtool.app) エクスポート (.json)
  - [SExtractor](https://github.com/satan53x/SExtractor) エクスポート (.txt .json .xlsx)
  - [VNTextPatch](https://github.com/arcusmaximus/VNTranslationTools) exports (.json)
  - [Translator++](https://dreamsavior.net/translator-plusplus) プロジェクト (.trans)
  - [Translator++](https://dreamsavior.net/translator-plusplus) エクスポート (.xlsx)
- 例については [Wiki - 対応フォーマット](https://github.com/neavo/LinguaGacha/wiki/%E6%94%AF%E6%8C%81%E7%9A%84%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F) を参照。フォーマットのリクエストは [ISSUES](https://github.com/neavo/LinguaGacha/issues) で提出

## FAQ 📥
- [LinguaGacha](https://github.com/neavo/LinguaGacha) と [AiNiee](https://github.com/NEKOparapa/AiNiee) の関係
  - `LinguaGacha` の作者は `AiNiee v5` の主要な開発およびメンテナンス担当者の一人です
  - `LinguaGacha` は `AiNiee` の派生ではなく、`AiNiee` の経験を基に開発された、よりシンプルで強力な、全く新しい翻訳アプリです
  - よりシンプルに、よりパワフルに。機能の強力さは、煩雑な設定を必要としません。

## サポート 😥
- 実行時のログは `log` フォルダに保存されます
- 問題を報告する際は、関連するログを添付してください
- グループに参加して、ディスカッションやフィードバックもできます。
  - Discord - https://discord.gg/pyMRBGse75