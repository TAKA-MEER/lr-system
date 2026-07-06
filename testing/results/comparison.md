# 実行比較

| run_id | STT CER | 話者精度 | 協議事項再現率 | アクション再現率 | 矛盾解消OK | Stage2失敗回数 |
|---|---|---|---|---|---|---|
| baseline_01 | 0.316 | 0.557 | 0.000 | 0.000 | False | 1 |
| iter1_temperature | 0.316 | 0.604 | 0.857 | 0.333 | False | 1 |
| iter2_topic_merge | 0.316 | 0.726 | 0.857 | 0.667 | True | 0 |
| iter3_action_completeness | 0.316 | 0.726 | 0.714 | 1.000 | True | 0 |
| iter3_repeat_check | 0.316 | 0.679 | 0.857 | 0.667 | True | 0 |
| iter4_stt_header_fix | 0.151 | 0.708 | 0.857 | 1.000 | True | 0 |
| iter5_vram_unload_fix | 0.151 | 0.708 | 0.857 | 1.000 | True | 0 |
| iter6_qwen3_8b | 0.151 | 0.660 | 0.000 | 0.000 | False | 1 |
| iter6b_qwen3_8b_nothink | 0.151 | 0.604 | 1.000 | 1.000 | False | 0 |
| iter6c_qwen3_8b_repeat | 0.151 | 0.745 | 0.857 | 1.000 | False | 0 |
| iter7_qwen3_5_9b | 0.151 | 0.660 | 1.000 | 1.000 | True | 0 |
| iter7b_qwen3_5_9b_repeat | 0.151 | 0.604 | 1.000 | 1.000 | True | 0 |
