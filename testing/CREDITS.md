# VOICEVOXクレジット表記

`testing/`配下のシミュレーション試験（立会試験を模した台本の音声合成、
`testing/tts/generate_audio.py` / `testing/tts/voice_map.yaml`）では、
[VOICEVOX](https://voicevox.hiroshiba.jp/)を用いて検証用の合成音声を作成している。
本番のLRシステム（`app/`配下）自体はVOICEVOXを使用しない（音声合成は行わず、
文字起こし・議事録生成のみを行う）。

VOICEVOXソフトウェア利用規約により、生成音声を利用する際は「VOICEVOXを利用した
ことがわかるクレジット表記」に加え、使用した各音声ライブラリ（キャラクター）ごとの
利用規約に従う必要がある（詳細: <https://voicevox.hiroshiba.jp/term/>）。

## 使用キャラクターとクレジット表記

`voice_map.yaml`で使用している4キャラクター。クレジット表記・規約URLは
[VOICEVOX/voicevox_vvm README](https://github.com/VOICEVOX/voicevox_vvm/blob/main/README.md)
（2026年9月時点）による。

| 話者(台本上の役) | VOICEVOXキャラクター | クレジット表記 | 利用規約 | 備考 |
|---|---|---|---|---|
| 山田 隆（client） | 青山龍星 | `VOICEVOX:青山龍星` | https://www.virvoxproject.com/voicevoxの利用規約 | **企業が携わる形で利用する場合は「[ななはぴ](https://v.seventhh.com/contact/)」への事前確認が必要**（個人利用は確認不要） |
| 佐藤 健二（client） | 玄野武宏 | `VOICEVOX:玄野武宏` | https://www.virvoxproject.com/voicevoxの利用規約 | 商用・非商用とも制限なく利用可 |
| 田中 誠（our_side） | 剣崎雌雄 | `VOICEVOX:剣崎雌雄` | https://frontier.creatia.cc/fanclubs/413/posts/4507 | 商用・非商用とも制限なく利用可 |
| 鈴木 舞（our_side） | 波音リツ | `VOICEVOX:波音リツ` | http://canon-voice.com/kiyaku.html | 商用・非商用とも制限なく利用可 |

## 本試験での用途とクレジット

上記4キャラクターの音声は、配電盤立会試験を模した台本（`scenario/script_v1.yaml`,
`scenario/script_endurance.yaml`）の読み上げにのみ使用しており、生成音声そのものを
公開・配布する用途ではなく、LRシステムのSTT/LLMパイプラインへの入力（検証用データ）
として一時的に使用している。それでも、検証結果を含む資料（例:
`reports/mirsdoc2/test0003a/`のPoC検証資料）で音声合成の方式に言及・スクリーンショット
等で言及する場合は、以下のクレジットを記載すること。

```
VOICEVOX:青山龍星、VOICEVOX:玄野武宏、VOICEVOX:剣崎雌雄、VOICEVOX:波音リツ
```

**青山龍星の利用について**: 本プロジェクトは学校（MIRS2602班）の開発物であり、
音声合成自体は学内検証目的で個人（開発担当者）が行っているが、検証結果は共同開発先
である明電舎（企業）への説明・提出に使われる可能性がある。これが上記規約にいう
「企業が携わる形での利用」に該当するか（＝ななはぴへの事前確認が必要か）は資料の
提出方法・用途によって判断が分かれるため、明電舎向けに本資料を正式に提出する前に、相談のうえ、必要であれば
[ななはぴへの問い合わせフォーム](https://v.seventhh.com/contact/)で事前確認を取ること。
