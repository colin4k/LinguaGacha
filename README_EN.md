<div align=center><img src="https://github.com/user-attachments/assets/cdf990fb-cf03-4370-a402-844f87b2fab8" width="256px;"></div>
<div align=center><img src="https://img.shields.io/github/v/release/neavo/LinguaGacha"/>   <img src="https://img.shields.io/github/license/neavo/LinguaGacha"/>   <img src="https://img.shields.io/github/stars/neavo/LinguaGacha"/></div>
<p align='center'>Next-generation text translator utilizing AI capabilities for one-click translation of novels, games, subtitles, and more</p>

&ensp;
&ensp;

## README 🌍
- [ [中文](./README.md) ] | [ [English](./README_EN.md) ] | [ [日本語](./README_JA.md) ]

## Overview 📢
- [LinguaGacha](https://github.com/neavo/LinguaGacha) (/ˈlɪŋɡwə ˈɡɑːtʃə/), abbreviated as `LG`, is an AI-powered next-generation text translator
- Out of the box, (almost) no setup needed, powerful does not need to be shown through complicated setting options
- Supports one-click translation between 13 languages
  - including `Chinese`, `English`, `Japanese`, `Korean`, `Russian`, `German`, `French`, `Italian`, etc
- Supports various text types and formats such as `subtitles`, `e-books`, and `game text`
- Supports both local and online interfaces such as `Claude`, `ChatGPT`, `DeepSeek`, `SakuraLLM`

> <img src="https://github.com/user-attachments/assets/859a7e32-bf35-4572-8460-4ecb11a8d20c" style="width: 80%;">

> <img src="https://github.com/user-attachments/assets/c0d7e898-f6fa-432f-a3cd-e231b657c4b5" style="width: 80%;">

## Special Notice ⚠️
- If you use [LinguaGacha](https://github.com/neavo/LinguaGacha) during translation, please include clear attribution in prominent locations of your work's information or release pages!
- For projects involving commercial activities or profits, please contact the author for authorization before using [LinguaGacha](https://github.com/neavo/LinguaGacha)!

## Key Features 📌
- Lightning-fast translation speed: 10 seconds for subtitles, 1 minute for novels, 5 minutes for games
- Automatic glossary generation ensuring consistent terminology (e.g., character names) throughout the work `👈👈 Exclusive Feature`
- Optimal translation quality from flagship models (e.g., DeepSeek-R1) to local small models (e.g., Qwen2.5-7B)
- `100%` accurate preservation of text formatting and embedded codes, significantly reducing post-processing work - ideal for embedded localization `👈👈 Exclusive Feature`

## System Requirements 🖥️
- Compatible with AI model interfaces following `OpenAI`, `Google`, `Anthropic`, `SakuraLLM` standards
- Compatible with [KeywordGacha](https://github.com/neavo/KeywordGacha) `👈👈 Next-generation tool for AI-powered glossary generation`

## Basic Workflow 🛸
- Download application from [Releases page](https://github.com/neavo/LinguaGacha/releases)
- Obtain a reliable AI model interface (choose one):
  - [ [Local API](https://github.com/neavo/OneClickLLAMA) ] (Free, requires ≥8GB VRAM GPU, Nvidia recommended)
  - [ [DeepSeek API](https://github.com/neavo/LinguaGacha/wiki/DeepSeek) ] (Paid, cost-effective, fast, high-quality, no GPU required) `👈👈 Recommended`
  - [ [Google Gemini API](https://aistudio.google.com/) ] (Paid, cost-effective, fast, relatively-high-quality, no GPU required) `👈👈 Recommended`
- Prepare source text:
  - `Subtitles`/`E-books` typically require no preprocessing
  - `Game texts` need extraction using appropriate tools for specific game engines
- Launch application via `app.exe`:
  - Configure essential settings (source/target languages) in `Project Settings`
  - Copy files to input folder (default: `input`), start translation in `Begin Translation`

## User Guide 📝
- Text Tutorial
  - [Basic Tutorial](https://github.com/neavo/LinguaGacha/wiki/BasicTutorial)　`👈👈 Step-by-step tutorial, easy to follow, a must-read for beginners`
- Video Tutorial
  - [How to Translate RPGMV with LinguaGacha and Translator++ (English)](https://www.youtube.com/watch?v=wtV_IODzi8I)
- Advance Tutorial
  - [Glossary](https://github.com/neavo/LinguaGacha/wiki/Glossary-%E2%80%90-EN)　　[Replacement](https://github.com/neavo/LinguaGacha/wiki/Replacement-%E2%80%90-EN)　　[MTool Optimizer](https://github.com/neavo/LinguaGacha/wiki/MToolOptimizer-%E2%80%90-EN)
  - [Best Practices for High-Quality Translation of RPGMaker Series Engine Games](https://github.com/neavo/LinguaGacha/wiki/BestPracticeForRPGMaker-%E2%80%90-EN)
- You can find more details on each feature in the [Wiki](https://github.com/neavo/LinguaGacha/wiki), and you are welcome to share your experience in the [Discussions](https://github.com/neavo/LinguaGacha/discussions)

## Supported Formats 🏷️
- Processes all supported files in input folder (including subdirectories):
  - Subtitles (.srt .ass)
  - E-books (.txt .epub)
  - Markdown（.md）
  - [RenPy](https://www.renpy.org) exports (.rpy)
  - [MTool](https://mtool.app) exports (.json)
  - [SExtractor](https://github.com/satan53x/SExtractor) exports (.txt .json .xlsx)
  - [Translator++](https://dreamsavior.net/translator-plusplus) project (.trans)
  - [Translator++](https://dreamsavior.net/translator-plusplus) exports (.xlsx)
- See [Wiki - Supported Formats](https://github.com/neavo/LinguaGacha/wiki/%E6%94%AF%E6%8C%81%E7%9A%84%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F) for examples. Submit format requests via [ISSUES](https://github.com/neavo/LinguaGacha/issues)

## Recent Updates 📅
- 20250316 v0.14.0
  - Translation Quality Focus Release
    - ADD - Result check now detects residual `Kana` and `Hangul`
    - OPT - Further enhanced code preservation
    - OPT - Significantly improved translation quality, especially for interjections and proper nouns
      - Highly recommended for all Volcengine DeepSeek users

- 20250316 v0.13.2
  - FIX - Some game engine types in the TRANS file could not be recognized

- 20250316 v0.13.1
  - OPT - Added some new cleanup rules
    - Annotation code `<Ruby>`
  - OPT - Some re-translation support for RENPY files
  - FIX - Some cases where EPUB wasn't outputting bilingual text correctly during translation

- 20250315 v0.13.0
  - NEW - Partial Retranslation Feature
    - Retranslates parts of completed translations based on set filter conditions
    - Primarily for content updates or bug fixes in subtitles, ebooks, etc

- 20250313 v0.12.3
  - OPT - If a data entry in the .trans file has a AQUA tag, force re-translation
  - FIX - Compatibility issues on some .trans files

- 20250312 v0.12.2
  - OPT - Performance overhead is way down when there's a lot of concurrent tasks (>=128)
  - OPT - When translating .trans files, it's more granular now, to tell apart identical text entries from different addresses
  - FIX - Fixed some problems that could cause translation to get stuck

## FAQ 📥
- Relationship between [LinguaGacha](https://github.com/neavo/LinguaGacha) and [AiNiee](https://github.com/NEKOparapa/AiNiee)
  - `LinguaGacha`'s developer was a main contributor to `AiNiee v5`
  - `LinguaGacha` is not a fork version of `AiNiee`, it's a new app built upon the lessons learned from `AiNiee`
  - It's simpler, and it's more powerful. Powerful does not need to be shown through complicated setting options

## Support 😥
- Runtime logs are stored in `log` folder
- Please attach relevant logs when reporting issues
- You can also join our groups for discussion and feedback:
  - Discord - https://discord.gg/pyMRBGse75
