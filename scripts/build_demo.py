"""把采样结果打包成一个可离线打开的静态 demo 页。

不加载模型、不调后端：读取 run_sampling.py 产出的 JSONL，前端只做展示，
判分与聚合全部在这里算好写进 HTML。用于面试现场演示「8 条候选 → 投票 → 一致度 → 转人工」。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402
from math_posttrain.inference_time import majority_vote  # noqa: E402


def build_record(row: dict, dataset: str, greedy_ok: bool | None) -> dict:
    candidates = row["candidates"]
    values = [extract_final_answer(text, dataset) for text in candidates]
    counts = Counter(value for value in values if value is not None)
    valid = sum(counts.values())
    top = max(counts.values()) if counts else 0
    share = top / valid if valid else 0.0
    chosen = majority_vote(candidates, dataset)
    wrapped = f"#### {chosen}" if dataset == "gsm8k" else f"\\boxed{{{chosen}}}"
    vote_ok = chosen is not None and is_correct(wrapped, row["reference"], dataset)
    return {
        "id": row["id"],
        "question": row["question"],
        "reference": row["reference"],
        "chosen": chosen,
        "share": round(share, 3),
        "votes": dict(counts.most_common()),
        "valid": valid,
        "vote_ok": vote_ok,
        "greedy_ok": greedy_ok,
        "route": "auto" if share >= 0.75 else "human",
        "candidates": [
            {"text": text, "answer": value} for text, value in zip(candidates, values)
        ],
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>推理时验证 demo · GSM8K dev 200</title>
<style>
:root {
  --bg: #f7f8fa; --panel: #ffffff; --ink: #1b1f24; --muted: #667085;
  --line: #e4e7ec; --ok: #1e8e5a; --bad: #c0392b; --accent: #2f6fed; --warn: #b26a00;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
header { padding: 18px 24px 12px; border-bottom: 1px solid var(--line); background: var(--panel); }
h1 { margin: 0 0 6px; font-size: 17px; font-weight: 650; }
.sub { color: var(--muted); font-size: 12.5px; }
.stats { display: flex; gap: 22px; margin-top: 12px; flex-wrap: wrap; }
.stat b { display: block; font-size: 19px; font-weight: 650; }
.stat span { color: var(--muted); font-size: 11.5px; }
main { display: grid; grid-template-columns: 300px 1fr; gap: 0; height: calc(100vh - 108px); }
#list { overflow-y: auto; border-right: 1px solid var(--line); background: var(--panel); }
.item { padding: 9px 14px; border-bottom: 1px solid var(--line); cursor: pointer; }
.item:hover { background: #f0f4ff; }
.item.active { background: #e8efff; border-left: 3px solid var(--accent); padding-left: 11px; }
.item .q { font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.item .meta { font-size: 11px; color: var(--muted); display: flex; gap: 8px; margin-top: 2px; }
#detail { overflow-y: auto; padding: 20px 24px 40px; }
.qtext { font-size: 15px; line-height: 1.65; margin-bottom: 14px; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.card .k { color: var(--muted); font-size: 11.5px; }
.card .v { font-size: 17px; font-weight: 650; margin-top: 2px; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11.5px; border: 1px solid; }
.tag.ok { color: var(--ok); border-color: #bfe0cd; background: #f0f9f4; }
.tag.bad { color: var(--bad); border-color: #f2c4c0; background: #fdf1f0; }
.tag.auto { color: var(--accent); border-color: #c6d8ff; background: #eef4ff; }
.tag.human { color: var(--warn); border-color: #f0dcb8; background: #fdf6e9; }
h3 { font-size: 13px; margin: 20px 0 8px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.cand { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 8px; }
.cand .head { display: flex; justify-content: space-between; gap: 10px; padding: 7px 12px;
  border-bottom: 1px solid var(--line); font-size: 12px; color: var(--muted); }
.cand pre { margin: 0; padding: 10px 12px; white-space: pre-wrap; word-break: break-word;
  font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; color: #2b3138; max-height: 190px; overflow: auto; }
.cand.match { border-color: #c6d8ff; }
.bar { height: 6px; background: #eef0f4; border-radius: 999px; overflow: hidden; margin-top: 6px; }
.bar i { display: block; height: 100%; background: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>推理时验证 demo · 模型冻结，只加采样与投票</h1>
  <div class="sub">Qwen3-4B-Base × GSM8K dev 前 200 题 × k=8 × T=0.7 · 采样耗时 52 分钟 / 1600 条候选 · 页面离线可开</div>
  <div class="stats">
    <div class="stat"><b>0.770</b><span>greedy pass@1（E0）</span></div>
    <div class="stat"><b>0.920</b><span>maj@8（投票后）</span></div>
    <div class="stat"><b>0.965</b><span>oracle pass@8（上界）</span></div>
    <div class="stat"><b>31 / 1</b><span>救回 / 带偏</span></div>
    <div class="stat"><b>84%</b><span>一致度 ≥75% 可自动批改（97.6% 正确）</span></div>
  </div>
</header>
<main>
  <div id="list"></div>
  <div id="detail"><p class="sub">从左侧选一道题查看 8 条候选、投票结果与「自动批改 / 转人工」判定。</p></div>
</main>
<script>
const DATA = __DATA__;
const list = document.getElementById('list');
const detail = document.getElementById('detail');
const esc = (s) => s.replace(/[&<>]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function renderList() {
  list.innerHTML = DATA.map((row, i) => {
    const tag = row.vote_ok ? '<span class="tag ok">对</span>' : '<span class="tag bad">错</span>';
    const route = row.route === 'auto' ? '自动' : '转人工';
    return `<div class="item" data-i="${i}">
      <div class="q">${esc(row.question.slice(0, 46))}</div>
      <div class="meta">${tag}<span>一致度 ${(row.share*100).toFixed(0)}%</span><span>${route}</span></div>
    </div>`;
  }).join('');
  [...list.children].forEach((el) => el.onclick = () => select(+el.dataset.i));
}

function select(i) {
  [...list.children].forEach((el) => el.classList.toggle('active', +el.dataset.i === i));
  const row = DATA[i];
  const voteTag = row.vote_ok ? '<span class="tag ok">投票正确</span>' : '<span class="tag bad">投票错误</span>';
  const greedyTag = row.greedy_ok === null ? '' :
    (row.greedy_ok ? '<span class="tag ok">greedy 对</span>' : '<span class="tag bad">greedy 错</span>');
  const routeTag = row.route === 'auto'
    ? '<span class="tag auto">一致度 ≥75%，自动批改</span>'
    : '<span class="tag human">一致度 <75%，建议转人工</span>';
  const votes = Object.entries(row.votes).map(([k, v]) =>
    `<span class="tag ${k === row.chosen ? 'ok' : ''}">${esc(k)} × ${v}</span>`).join(' ');
  detail.innerHTML = `
    <div class="qtext">${esc(row.question)}</div>
    <div class="summary">
      <div class="card"><div class="k">参考答案（公开答案）</div><div class="v">${esc(row.reference)}</div></div>
      <div class="card"><div class="k">投票结果</div><div class="v">${row.chosen === null ? '—' : esc(row.chosen)}</div>
        <div class="bar"><i style="width:${row.share*100}%"></i></div></div>
      <div class="card"><div class="k">一致度 / 有效候选</div><div class="v">${(row.share*100).toFixed(0)}% · ${row.valid}/8</div></div>
      <div class="card"><div class="k">结论</div><div class="v" style="font-size:13px">${voteTag} ${greedyTag}</div>
        <div style="margin-top:6px">${routeTag}</div></div>
    </div>
    <div class="sub" style="margin-bottom:4px">票数分布：${votes || '—'}</div>
    <h3>8 条候选（抽取到的答案标在右上角）</h3>
    ${row.candidates.map((c, j) => `
      <div class="cand ${c.answer !== null && c.answer === row.chosen ? 'match' : ''}">
        <div class="head"><span>候选 ${j + 1}</span><span>${c.answer === null ? '抽取失败' : '答案 ' + esc(c.answer)}</span></div>
        <pre>${esc(c.text)}</pre>
      </div>`).join('')}
  `;
  detail.scrollTop = 0;
}

renderList();
select(0);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-file", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("demo/index.html"))
    args = parser.parse_args()

    greed: dict[str, bool] = {}
    if args.baseline:
        for line in args.baseline.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                greed[row["id"]] = is_correct(row["prediction"], row["reference"], args.dataset)

    rows = [json.loads(line) for line in args.sampling_file.open(encoding="utf-8") if line.strip()]
    records = [build_record(row, args.dataset, greed.get(row["id"])) for row in rows]

    # 选样：优先展示「救回」「转人工」「投票错误」三类，便于演示
    def priority(record: dict) -> tuple:
        rescued = record["greedy_ok"] is False and record["vote_ok"]
        return (not rescued, record["route"] == "auto", record["vote_ok"])

    picked = sorted(records, key=priority)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        TEMPLATE.replace("__DATA__", json.dumps(picked, ensure_ascii=False)),
        encoding="utf-8",
    )
    auto = sum(1 for record in picked if record["route"] == "auto")
    print(json.dumps({
        "written": str(args.output),
        "questions": len(picked),
        "auto_route": auto,
        "human_route": len(picked) - auto,
        "rescued_shown": sum(1 for r in picked if r["greedy_ok"] is False and r["vote_ok"]),
        "size_kb": round(args.output.stat().st_size / 1024, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
