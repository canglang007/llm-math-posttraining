"""生成可交互的逐题对比 demo（单文件 HTML，数据全部来自真实 test 预测）。

选题策略（不是手工挑，而是按规则抽）：
- A2（不验证）与 E2b（验证）分叉题：各方向取前 N 题
- SFT 与 DPO(26 对) 分叉题：各方向取前 N 题
- Base 抽取失败样例：取前 N 题（说明"不会收尾"）

用法：
    python scripts/build_comparison_demo.py --out demo/comparison.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402

MODELS = [
    ("base", "Base"),
    ("sft", "SFT"),
    ("e2b-rft300q", "E2b 验证"),
    ("a2-nofilter", "A2 不验证"),
    ("e3b-dpo-from-sft", "DPO 26 对"),
]
TEST_STATS = [
    ("E0 Base", 0.7604, 1003, 175, "—"),
    ("E1 SFT", 0.8112, 1070, 3, "基准"),
    ("E2a RFT-114q", 0.8089, 1067, 1, "−0.23pt"),
    ("E2b RFT-286q（验证）", 0.8120, 1071, 0, "+0.08pt"),
    ("E2c RFT-from-Base", 0.8074, 1065, 53, "−0.38pt"),
    ("A2 RFT（不验证）", 0.7945, 1048, 0, "−1.67pt"),
    ("E3a DPO（RFT 起）", 0.8014, 1057, 1, "−0.99pt"),
    ("E3b DPO（26 对）", 0.8203, 1082, 2, "+0.91pt"),
    ("E3c DPO（79 对）", 0.8188, 1080, 3, "+0.76pt"),
    ("E4 GRPO", 0.8089, 1067, 2, "−0.23pt"),
    ("E4b GRPO（难题 26）", 0.8112, 1070, 1, "0.00pt"),
]


def load(path: Path) -> dict[str, dict]:
    return {str(json.loads(line)["id"]): json.loads(line) for line in path.open()}


def graded(rows: dict[str, dict]) -> dict[str, bool]:
    return {qid: is_correct(r.get("prediction", ""), r.get("reference") or "", "gsm8k") for qid, r in rows.items()}


def pick(runs: dict[str, dict], a_key: str, b_key: str, limit: int) -> list[tuple[str, str]]:
    a, b = graded(runs[a_key]), graded(runs[b_key])
    a_only, b_only = [], []
    for qid in sorted(set(a) & set(b)):
        if a[qid] and not b[qid]:
            a_only.append((qid, "a_only"))
        elif b[qid] and not a[qid]:
            b_only.append((qid, "b_only"))
    return a_only[:limit] + b_only[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("experiments/eval/vllm-test"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/gsm8k/test.jsonl"))
    parser.add_argument("--per-direction", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("demo/comparison.html"))
    args = parser.parse_args()

    runs = {key: load(args.eval_dir / f"{key}.jsonl") for key, _ in MODELS}
    questions = {str(json.loads(line)["id"]): json.loads(line)["question"] for line in args.questions.open()}

    groups: list[dict] = []

    def add_group(title: str, note: str, pairs: list[tuple[str, str]], tag: str) -> None:
        items = []
        for qid, direction in pairs:
            answers = {}
            for key, label in MODELS:
                row = runs[key][qid]
                answers[key] = {
                    "label": label,
                    "text": row.get("prediction", ""),
                    "ok": is_correct(row.get("prediction", ""), row.get("reference") or "", "gsm8k"),
                    "answer": extract_final_answer(row.get("prediction", ""), "gsm8k"),
                }
            items.append(
                {
                    "id": qid,
                    "question": questions.get(qid, ""),
                    "reference": runs["sft"][qid].get("reference") or "",
                    "direction": direction,
                    "answers": answers,
                }
            )
        groups.append({"title": title, "note": note, "tag": tag, "items": items})

    add_group(
        "A2 消融：E2b（验证）对、A2（不验证）错",
        "同一批候选、同题量、同起点，唯一差别是 37 条错误解是否过滤。",
        pick(runs, "e2b-rft300q", "a2-nofilter", args.per_direction)[: args.per_direction],
        "verify",
    )
    add_group(
        "A2 消融：A2 对、E2b 错",
        "反方向也存在——验证不是万能的，只是统计上更好（+1.74pt，p=0.022）。",
        pick(runs, "e2b-rft300q", "a2-nofilter", args.per_direction)[args.per_direction :],
        "verify",
    )
    add_group(
        "DPO(26 对) 对、SFT 错",
        "偏好数据（外部答案验证信号）改变了什么：多为题目关系建模，而非算术。",
        pick(runs, "sft", "e3b-dpo-from-sft", args.per_direction)[args.per_direction :],
        "dpo",
    )
    add_group(
        "SFT 对、DPO(26 对) 错",
        "DPO 的反向代价：偏好数据也会带来新的错误模式。",
        pick(runs, "sft", "e3b-dpo-from-sft", args.per_direction)[: args.per_direction],
        "dpo",
    )

    payload = {
        "stats": [
            {"name": n, "acc": a, "correct": c, "fail": f, "delta": d} for n, a, c, f, d in TEST_STATS
        ],
        "models": [{"key": k, "label": v} for k, v in MODELS],
        "groups": groups,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"written: {args.out}  groups={len(groups)} questions={sum(len(g['items']) for g in groups)}")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LLM Math Post-Training · 逐题对比</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --bg:#f7f7f5; --panel:#ffffff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e4e0;
    --ok:#1f7a4d; --ok-bg:#eaf5ee; --bad:#b03030; --bad-bg:#fbebeb; --accent:#2d5f8b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",Arial,sans-serif;
       line-height:1.6;-webkit-font-smoothing:antialiased}
  header{padding:32px 40px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{margin:0 0 6px;font-size:22px;letter-spacing:.2px}
  .sub{color:var(--muted);font-size:13px}
  .wrap{display:grid;grid-template-columns:minmax(360px,420px) 1fr;gap:0;min-height:calc(100vh - 96px)}
  aside{border-right:1px solid var(--line);background:var(--panel);padding:16px 0 40px;overflow-y:auto;max-height:calc(100vh - 96px)}
  main{padding:24px 32px 60px;overflow-y:auto;max-height:calc(100vh - 96px)}
  table{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0 18px}
  th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.4px}
  tr.hl td{background:#fbfaf4}
  .num{font-variant-numeric:tabular-nums;text-align:right}
  .tag{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:10px;border:1px solid var(--line);color:var(--muted);margin-right:6px}
  .gtitle{font-size:12px;color:var(--muted);padding:14px 16px 6px;letter-spacing:.3px}
  .qitem{padding:9px 16px;border-left:3px solid transparent;cursor:pointer;font-size:12.5px}
  .qitem:hover{background:#f4f4f1}
  .qitem.active{border-left-color:var(--accent);background:#f0f4f8}
  .qitem .qid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--muted)}
  .qcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-bottom:18px}
  .qtext{font-size:14px;margin:6px 0 14px}
  .ans{border-top:1px dashed var(--line);padding:12px 0 4px}
  .ans:first-of-type{border-top:none}
  .anshead{display:flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600;margin-bottom:4px}
  .badge{font-size:10.5px;padding:1px 6px;border-radius:6px;font-weight:600}
  .badge.ok{color:var(--ok);background:var(--ok-bg)}
  .badge.bad{color:var(--bad);background:var(--bad-bg)}
  .badge.ans{color:var(--muted);background:#f2f2ef;font-family:ui-monospace,monospace}
  pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
      font-size:12px;line-height:1.55;color:#2b2b2b;background:#fafaf8;border:1px solid #efefea;border-radius:8px;padding:9px 11px;max-height:190px;overflow:auto}
  .hint{color:var(--muted);font-size:12px;margin:2px 0 14px}
  .empty{color:var(--muted);padding:40px 0}
  .dirnote{font-size:11px;color:var(--muted);margin-top:4px}
</style>
</head>
<body>
<header>
  <h1>LLM Math Post-Training · 逐题对比</h1>
  <div class="sub">Qwen3-4B-Base × GSM8K test 1319 题 · vLLM 0.34 + bf16 + greedy · 数据全部来自真实预测 jsonl</div>
</header>
<div class="wrap">
  <aside id="side"></aside>
  <main id="main"></main>
</div>
<script>
const DATA = __DATA__;
const side = document.getElementById('side');
const main = document.getElementById('main');
let active = null;

function pct(x){return (x*100).toFixed(2)+'%';}

function renderStats(){
  let h = '<div class="gtitle">test 1319 题 · 11 档（权威口径）</div>';
  h += '<table><tr><th>模型</th><th class="num">准确率</th><th class="num">抽取失败</th><th>vs SFT</th></tr>';
  for(const s of DATA.stats){
    const hl = (s.name.includes('SFT')||s.name.includes('A2')) ? ' class="hl"' : '';
    h += `<tr${hl}><td>${s.name}</td><td class="num">${pct(s.acc)}</td><td class="num">${s.fail}</td><td>${s.delta}</td></tr>`;
  }
  h += '</table>';
  return h;
}

function renderSide(){
  let h = renderStats();
  DATA.groups.forEach((g, gi) => {
    h += `<div class="gtitle">${g.title}<div class="dirnote">${g.note}</div></div>`;
    g.items.forEach((it, ii) => {
      h += `<div class="qitem" data-g="${gi}" data-i="${ii}">
              <div class="qid">${it.id} · 参考答案 ${it.reference}</div>
              <div>${it.question.slice(0,52)}${it.question.length>52?'…':''}</div>
            </div>`;
    });
  });
  side.innerHTML = h;
  side.querySelectorAll('.qitem').forEach(el => el.onclick = () => {
    side.querySelectorAll('.qitem').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    const g = DATA.groups[+el.dataset.g], it = g.items[+el.dataset.i];
    renderQuestion(g, it);
  });
  side.querySelector('.qitem')?.click();
}

function renderQuestion(g, it){
  const order = ['sft','e2b-rft300q','a2-nofilter','e3b-dpo-from-sft','base'];
  let h = `<div class="qcard"><div><span class="tag">${g.tag === 'verify' ? 'A2 消融' : 'DPO 对比'}</span><span class="qid">${it.id}</span></div>`;
  h += `<div class="qtext">${it.question.replace(/\\n/g,'<br>')}</div>`;
  h += `<div class="hint">参考答案：<b>${it.reference}</b> · 下面五列是各模型自己的完整输出（截断显示，可滚动）</div>`;
  for(const key of order){
    const a = it.answers[key];
    h += `<div class="ans"><div class="anshead">${a.label}
            <span class="badge ${a.ok?'ok':'bad'}">${a.ok?'✓ 正确':'✗ 错误'}</span>
            <span class="badge ans">抽取答案 ${a.answer === null ? '无' : a.answer}</span></div>
          <pre>${escapeHtml(a.text)}</pre></div>`;
  }
  h += '</div>';
  main.innerHTML = h;
}

function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

renderSide();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
