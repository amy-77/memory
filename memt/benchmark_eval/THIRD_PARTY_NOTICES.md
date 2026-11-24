# Third-party code and prompts

## OmniMemEval (Apache-2.0)

`code/judge_bench.py` and `code/adapters.py` contain judge prompts and scoring rules copied verbatim from
OmniMemEval `scripts/utils/prompts.py` and its BEAM scorer (github.com/MemTensor/OmniMemEval, commit ff13e4a):
`JUDGE_PROMPT`, `JUDGE_SYSTEM_PROMPT`, the BEAM rubric-item and event-ordering judge prompts, item-score parsing and
Kendall tau-b. Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

NOTICE of OmniMemEval:

    OmniMemEval
    Copyright 2026 MemTensor

## MemoryAgentBench (MIT)

`code/mab_official.py` is copied verbatim from MemoryAgentBench (github.com/HUST-AI-HYZ/MemoryAgentBench,
commit 5380260): metric helpers from `utils/eval_other_utils.py`, judge prompts from `llm_based_eval/`, and the
templates of `utils/templates.py`.

    MIT License

    Copyright (c) 2026 Yuanzhe Hu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## MemT-1.0-Nano model code

`code/bench_nano_hf.py` loads MemTensor/MemT-1.0-Nano with `trust_remote_code` (the checkpoint's own
`modeling_memt.py`); no MemT code is copied here. The benchmark prompts follow appendix D.3 of the MemT-1.0
technical report (arXiv 2610.26825).
