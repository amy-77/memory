| Benchmark | MemT-Nano | MemT-Flash | QW full ctx | 4B memory (plain) | 4B question only | 4B wrong history | Qwen3-4B full ctx 32K | 4B memory (chat, MemT format) | 4B question only (chat) | MemT-Nano memory (ours, HF) | MemT-Nano empty memory (ours) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LoCoMo | 57.5 | 65.1 | 83.8 | 24.1 (n=1540) | 18.8 (n=1540) | 16.2 (n=1540) | 66.7 (n=1540) | 0.0 (n=382) | 21.2 (n=1540) | 51.7 (n=1540) | 2.7 (n=1540) |
| LongMemEval-S | 32.6 | 37.2 | 76.8 | 9.2 (n=500) | 12.2 (n=500) | 8.2 (n=500) | 27.2 (n=500) | 0.0 (n=500) | 8.0 (n=500) | 29.0 (n=500) | 7.4 (n=500) |
| PersonaMem-32K | 52.3 | 61.6 | 65.2 | 51.8 (n=589) | 22.4 (n=589) | 51.4 (n=589) | 48.9 (n=589) | - | - | 55.7 (n=589) | 44.8 (n=589) |
| MemoryAgentBench | 15.5 | 17.4 | 35.9 | 8.5 (n=2071) | 7.0 (n=2071) | 7.1 (n=2071) | 21.6 (n=2071) | - | - | 12.4 (n=2071) | - |
| BEAM-1M | 35.8 | 37.0 | 43.7 | 17.7 (n=700) | 19.4 (n=700) | 15.1 (n=700) | 18.4 (n=700) | - | - | 29.8 (n=700) | - |
