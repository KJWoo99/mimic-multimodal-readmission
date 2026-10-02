"""절제실험 기록들을 사람이 읽는 실험 일지로 만듦.

    python scripts/render_experiment_log.py

`outputs/ablation/*.json` 을 모아 `docs/EXPERIMENT_LOG.md` 를 씀.

## 회차마다 싣는 것

각 실험은 가설, 바꾼 것, 관찰, 판단을 함께 싣고, 부모 실험을 가리켜
어떤 결과를 보고 다음을 정했는지 보이게 함.

## 판정선을 기록에서 구함

판정선을 상수로 두면 시드만 바꿔도 그보다 크게 움직인다는 사실이 드러나지 않음(트러블슈팅 18).
그래서 판정선을 재현 회차(번호 90번대)의 흩어짐에서 계산함. 표본이 셋뿐이라
표준편차 자체가 불안정하므로 2배를 씀.

## 같은 시드끼리 짝지어 봄

시드의 영향이 설정과 무관하게 크면 평균끼리 비교할 때 시드 운이 그대로 섞여
들어감. 같은 시드끼리 짝지어 차이를 보면 그 운이 상쇄됨.

그래서 재현 회차가 있는 설정은 시드별 차이를 함께 싣고, 부호가 시드마다
갈리면 "방향이 일정하지 않다"고 적음.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_dataset import AUG_LEVELS as AUG_ORDER  # noqa: E402  약한 것부터 센 것 순

ABL_DIR = ROOT / "outputs" / "ablation"
OUT_MD = ROOT / "docs" / "EXPERIMENT_LOG.md"

# 90번 이상은 "설정을 바꾼 실험"이 아니라 "같은 설정을 시드만 바꿔 다시 돌린 회차"다.
# 절제 회차 수와 판정에서 따로 다룸.
REPEAT_ID_FROM = 90

# 재현 회차가 없을 때 쓸 값. 근거 없는 값이므로 일지에도 그렇게 적음.
FALLBACK_MEANINGFUL = 0.010



def _md_tilde(text: str) -> str:
    """GitHub 는 한 블록 안의 ~ 두 개를 취소선으로 그림. 코드 밖의 ~ 를 \\~ 로 적음."""
    out, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            parts = re.split(r"(`[^`]*`)", ln)
            ln = "".join(p if i % 2 else re.sub(r"(?<!\\)~", r"\\~", p) for i, p in enumerate(parts))
        out.append(ln)
    return "\n".join(out)

def _score(r: dict) -> float:
    return r["result"]["best_val_pr_auc"]


def _load() -> list[dict]:
    if not ABL_DIR.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ABL_DIR.glob("*.json"))]
    return sorted(recs, key=lambda r: r["id"])


def _fmt_line(m: float) -> str:
    """판정선 표기. 넷째 자리 반올림이 값을 바꾸면 반올림 전 값을 함께 적음."""
    return f"{m:.4f}" if round(m, 4) == m else f"{m:.4f}(반올림 전 {m:.6f})"


def _seed_map(recs: list[dict], trial_id: int) -> dict[int, float]:
    """설정 하나의 시드별 점수. 본 회차 + 그것을 되풀이한 회차."""
    out = {}
    for r in recs:
        if r["id"] == trial_id or (r["id"] >= REPEAT_ID_FROM and r.get("parent") == trial_id):
            out[r["config"]["seed"]] = _score(r)
    return out


def main() -> int:
    recs = _load()
    if not recs:
        print(f"기록이 없다: {ABL_DIR}")
        return 1

    trials = [r for r in recs if r["id"] < REPEAT_ID_FROM]
    repeats = [r for r in recs if r["id"] >= REPEAT_ID_FROM]
    base = next((r for r in trials if not r.get("parent")), trials[0])
    base_score = _score(base)
    base_seeds = _seed_map(recs, base["id"])

    if len(base_seeds) >= 3:
        sd = statistics.stdev(base_seeds.values())
        # 판정은 반올림하지 않은 2배 값으로 함(규칙이 2xSD 다). 표기만 넷째 자리로 줄임.
        meaningful = 2 * sd
        threshold_note = (
            f"판정선 {_fmt_line(meaningful)} 은 기준선을 시드만 바꿔 {len(base_seeds)}회 돌린 "
            f"결과의 표준편차({sd:.4f})의 2배다. 표본이 적어 표준편차가 불안정하므로 "
            "2배를 써서 보수적으로 잡았다.")
    else:
        meaningful = FALLBACK_MEANINGFUL
        threshold_note = (
            f"판정선 {meaningful:.4f} 은 근거 없이 정한 값이다. 기준선을 시드만 바꿔 "
            "세 번 이상 돌리면 실측으로 대체된다.")

    def verdict(delta: float | None) -> str:
        if delta is None:
            return "기준선"
        if delta >= meaningful:
            return f"개선 ({delta:+.4f})"
        if delta <= -meaningful:
            return f"악화 ({delta:+.4f})"
        return f"차이 없음 ({delta:+.4f})"

    lines: list[str] = []
    lines.append("# 실험 일지: 무엇을 바꿨더니 어떻게 됐는가\n")
    lines.append(
        "자동 생성 파일이다. `python scripts/render_experiment_log.py` 로 다시 만든다.\n")
    lines.append(
        "모든 수치는 val 기준이다. 탐색을 test 로 하면 열 몇 번 돌린 시행들의 운이\n"
        "test 점수에 섞여, 그 값이 더 이상 미래 성능의 추정치가 아니게 된다.\n"
        "test 는 최종 조합이 정해진 뒤 딱 한 번만 본다.\n")
    lines.append(f"기준선 = 실험 #{base['id']} ({base['name']}), "
                 f"val PR-AUC {base_score:.4f}\n")
    lines.append(f"> {threshold_note}\n")

    # ── 요약표 ────────────────────────────────────────────────────────
    lines.append("## 한눈에 보기\n")
    lines.append("| # | 바꾼 것 | val PR-AUC | 기준선 대비 | best/실행 | 진단 |")
    lines.append("|---|---|---:|---|---:|---|")
    for r in trials:
        d = None if r["id"] == base["id"] else _score(r) - base_score
        res = r["result"]
        lines.append(
            f"| {r['id']} | {r['change'] or ': '} | {_score(r):.4f} | "
            f"{verdict(d)} | {res['best_epoch']}/{res['epochs_run']} | {r['diagnosis']} |")
    lines.append("")

    # ── 시드 재현 ─────────────────────────────────────────────────────
    if repeats:
        lines.append("## 같은 설정을 시드만 바꿔 다시 돌린 회차\n")
        lines.append(
            "설정을 바꾼 실험이 아니다. 회차 간 차이가 어디까지 잡음인지 재려고\n"
            "돌린 것이므로 절제 회차 수에는 넣지 않는다.\n")
        parents = sorted({r["parent"] for r in repeats})
        seed_cols = sorted({r["config"]["seed"] for r in recs
                            if r["id"] in {*[p for p in parents]} or r["id"] >= REPEAT_ID_FROM
                            or r["id"] == base["id"]})
        lines.append("| 설정 | " + " | ".join(f"seed {s}" for s in seed_cols) + " | 평균 | 표준편차 |")
        lines.append("|---|" + "---:|" * (len(seed_cols) + 2))
        for pid in parents:
            src = next(r for r in trials if r["id"] == pid)
            m = _seed_map(recs, pid)
            vals = [m.get(s) for s in seed_cols]
            shown = " | ".join(f"{v:.4f}" if v is not None else ": " for v in vals)
            got = [v for v in vals if v is not None]
            sd = f"{statistics.stdev(got):.4f}" if len(got) > 1 else ": "
            lines.append(f"| #{pid} {src['name']} | {shown} | "
                         f"{statistics.mean(got):.4f} | {sd} |")
        lines.append("")

        # 같은 시드끼리 짝지은 차이
        others = [p for p in parents if p != base["id"]]
        if others:
            lines.append("### 같은 시드끼리 짝지은 차이\n")
            lines.append(
                "시드의 영향이 설정과 무관하게 크므로, 평균끼리 비교하면 시드 운이\n"
                "그대로 섞인다. 같은 시드끼리 빼면 그 운이 상쇄된다.\n")
            lines.append("| 설정 | " + " | ".join(f"seed {s}" for s in seed_cols) + " | 평균 차이 | 방향 |")
            lines.append("|---|" + "---:|" * (len(seed_cols) + 1) + "---|")
            for pid in others:
                src = next(r for r in trials if r["id"] == pid)
                m = _seed_map(recs, pid)
                diffs = {s: m[s] - base_seeds[s] for s in seed_cols
                         if s in m and s in base_seeds}
                if not diffs:
                    continue
                cells = " | ".join(f"{diffs[s]:+.4f}" if s in diffs else ": " for s in seed_cols)
                mean_d = statistics.mean(diffs.values())
                signs = {d > 0 for d in diffs.values()}
                direction = "일정" if len(signs) == 1 else "시드마다 갈린다"
                lines.append(f"| #{pid} {src['name']} | {cells} | {mean_d:+.4f} | {direction} |")
            lines.append("")

    # ── 실험별 경위 ───────────────────────────────────────────────────
    lines.append("## 실험별 경위\n")
    for r in trials:
        res, cfg = r["result"], r["config"]
        d = None if r["id"] == base["id"] else _score(r) - base_score
        lines.append(f"### 실험 #{r['id']}: {r['name']}\n")
        if r.get("parent"):
            lines.append(f"- 부모 회차: 실험 #{r['parent']}")
        if r.get("hypothesis"):
            lines.append(f"- 가설: {r['hypothesis']}")
        lines.append(f"- 바꾼 것: {r['change'] or '없음(기준선)'}")
        lines.append(
            f"- 결과: val PR-AUC {_score(r):.4f} "
            f"({verdict(d)}), best {res['best_epoch']}에폭 / 실행 {res['epochs_run']}에폭")
        m = _seed_map(recs, r["id"])
        if len(m) > 1:
            lines.append("- 시드별: " + ", ".join(f"seed {s} {v:.4f}" for s, v in sorted(m.items())))
        lines.append(
            f"- 학습 곡선: train loss {res['first_train_loss']:.4f} -> "
            f"{res['final_train_loss']:.4f}, 조기종료 {res['stopped_early']}")
        lines.append(f"- 진단: {r['diagnosis']}")
        lines.append(
            f"- 설정: aug={cfg['aug']} lr={cfg['lr_head']:g}/{cfg['lr_backbone']:g} "
            f"wd={cfg['weight_decay']:g} batch={cfg['batch']} size={cfg['size']} "
            f"sched={cfg['scheduler']} pos_weight={cfg['pos_weight']}")
        lines.append("")

    # ── 결론 ──────────────────────────────────────────────────────────
    best = max(trials, key=_score)
    lines.append("## 여기까지의 결론\n")
    lines.append(
        f"- 가장 높았던 회차: 실험 #{best['id']} ({best['name']}) "
        f"val PR-AUC {_score(best):.4f}")
    improved = [r for r in trials if r["id"] != base["id"] and _score(r) - base_score >= meaningful]
    worsened = [r for r in trials if _score(r) - base_score <= -meaningful]
    lines.append(f"- 기준선보다 나아진 실험: {len(improved)}건"
                 + (f" {[r['name'] for r in improved]}" if improved else ""))
    lines.append(f"- 오히려 나빠진 실험: {len(worsened)}건"
                 + (f" {[r['name'] for r in worsened]}" if worsened else ""))
    # 설정 회차는 기준선 한 번 값과 비교함. 기준선 시드가 계열 안에서 높거나 낮으면 계열 평균과
    # 견줄 때 판정이 달라질 수 있어, 달라지는 회차를 함께 적음.
    if len(base_seeds) >= 3:
        fam_mean = statistics.mean(base_seeds.values())

        def _cat(delta: float) -> int:
            return 1 if delta >= meaningful else (-1 if delta <= -meaningful else 0)

        same_seed = all(r["config"]["seed"] == base["config"]["seed"] for r in trials)
        differ = [r for r in trials if r["id"] != base["id"]
                  and _cat(_score(r) - base_score) != _cat(_score(r) - fam_mean)]
        lines.append(
            f"- 판정은 {'같은 시드(seed ' + str(base['config']['seed']) + ')인 ' if same_seed else ''}"
            f"기준선 한 번 값({base_score:.4f})과 비교했다. 기준선 계열 평균({fam_mean:.4f})과 "
            f"비교하면 판정이 달라지는 실험: {len(differ)}건"
            + (" " + ", ".join(f"{r['name']}({_score(r) - fam_mean:+.4f})" for r in differ)
               if differ else ""))

    streak = 0
    for r in trials:
        if r["id"] <= 5 or r["id"] == base["id"]:
            continue
        streak = 0 if _score(r) - base_score >= meaningful else streak + 1
    lines.append(f"- 의무 구간(#1~5) 이후 연속 미개선: {streak}회 (3회면 종료)")
    lines.append(f"- 총 {len(trials)}회 실험 + 재현 {len(repeats)}회, 누적 "
                 f"{sum(r['elapsed_sec'] for r in recs) / 3600:.1f}시간")
    lines.append("")

    # ── 한계 ──────────────────────────────────────────────────────────
    # 수치를 본문에 손으로 적으면 실험이 늘 때 글만 낡음. 기록에서 뽑음.
    lines.append("## 이 실험의 한계\n")
    best_gap = _score(best) - base_score
    paired = any(r.get("parent") not in (None, 0, base["id"]) for r in repeats)
    if len(base_seeds) >= 3:
        spread = max(base_seeds.values()) - min(base_seeds.values())
        if best["id"] == base["id"]:
            # 승자가 기준선이면 "최대 차이 +0.0000" 처럼 뜻 없는 문장이 나옴.
            gaps = [_score(r) - base_score for r in trials if r["id"] != base["id"]]
            lines.append(
                f"기준선을 넘은 회차가 없다. 설정을 바꾼 {len(gaps)}회는 기준선보다 "
                f"{-max(gaps):.4f}~{-min(gaps):.4f} 낮았고, 같은 설정을 시드만 바꿨을 때 "
                f"{spread:.4f} 만큼 벌어진다. 가장 가까운 회차도 시드 폭 안이라 설정 차이로 "
                "읽지 않고, 기준선 설정을 그대로 확인 실행으로 넘긴다.\n")
        else:
            where = ("위의 짝 비교 표에 있다." if paired else
                     "아직 없다(기준선 말고는 시드를 여러 번 돌린 설정이 없다).")
            lines.append(
                f"설정을 바꿔 얻은 차이가 시드만 바꾼 폭보다 작다. 회차 간 최대 차이가 "
                f"{best_gap:+.4f} 인데, 같은 설정을 시드만 바꿨을 때 {spread:.4f} 만큼 벌어진다. "
                "그래서 이 표의 순위를 성능 개선으로 읽으면 안 된다. 어느 설정이 낫다고 "
                "말하려면 시드를 여러 개 돌려 같은 시드끼리 비교해야 하고, 그렇게 본 결과는 "
                f"{where}\n")
        lines.append(
            "이 값은 초기화, 배치순서에서 오는 폭만 담는다. 분할 시드는 42 로 고정했다. "
            "분할까지 바꾸면 앞 실행의 val 환자가 뒤 실행의 test 로 들어가기 때문이다. "
            "즉 실제 변동폭은 이보다 크면 컸지 작지 않다.\n")
    else:
        lines.append(
            f"시드 재현 회차가 아직 부족하다. 가장 좋았던 설정의 우위가 {best_gap:+.4f} 인데, "
            "이만한 차이가 잡음인지 가릴 근거가 이 실험 안에 없다.\n")

    # 증강 사다리: 한 축을 여러 단계로 재본 경우에만 적음.
    ladder = {}
    for r in trials:
        aug = r["config"]["aug"]
        if aug not in ladder or _score(r) > ladder[aug]:
            ladder[aug] = _score(r)
    steps = [(a, ladder[a]) for a in AUG_ORDER if a in ladder]
    if len(steps) >= 3:
        chain = " -> ".join(f"{a} {v:.4f}" for a, v in steps)
        peak = max(range(len(steps)), key=lambda i: steps[i][1])
        rising = all(steps[i][1] < steps[i + 1][1] for i in range(peak))
        if rising and peak < len(steps) - 1:
            # 이웃 단계 차이를 시드 폭과 견줌. 차이 일부가 시드 폭 안이면 "사다리 전체가 우연히 이 모양이
            # 되기는 어렵다" 고 말할 수 없음.
            gaps = [abs(steps[i + 1][1] - steps[i][1]) for i in range(len(steps) - 1)]
            seed_spread = (max(base_seeds.values()) - min(base_seeds.values())
                           if len(base_seeds) >= 3 else None)
            inside = sum(g < seed_spread for g in gaps) if seed_spread is not None else None
            if inside:
                why = (f"다만 이웃 단계 차이 {len(gaps)}개 중 {inside}개가 시드만 바꾼 폭"
                       f"({seed_spread:.4f}) 안이라 이 모양도 우연일 수 있다. 방향만 참고로 남기고 ")
            else:
                why = "이웃 단계 차이가 모두 시드 폭보다 커 방향은 남기되 "
            lines.append(
                f"증강 강도를 {len(steps)}단계로 재보니 {chain} 로, 오르다 마지막에 "
                f"꺾이는 모양이 나왔다. {why}'최적값이 어디다'는 주장은 하지 않는다.\n")

    lines.append(
        "통계 검정은 하지 않았다. 판정선은 검정이 아니라 회차 간 변동폭에서 뽑은 눈금이다. "
        "그 선을 넘었다고 유의하다는 뜻이 아니라, 그보다 작은 차이는 이 실험으로 구분할 수 "
        "없다는 뜻이다.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    # 줄끝은 LF 로 고정함. 플랫폼 기본값에 맡기면 CRLF 가 되어 표 한 줄만 바뀌어도 문서 전체가
    # 바뀐 것으로 나와 대조가 안 됨(.gitattributes 도 LF 로 정해 둠).
    OUT_MD.write_text(_md_tilde("\n".join(lines)), encoding="utf-8", newline="\n")
    print(f"[저장] {OUT_MD}  (실험 {len(trials)}건, 재현 {len(repeats)}건, "
          f"판정선 {meaningful:.4f})")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
