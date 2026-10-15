"""
ETF 특이 거래량 감시
- etfs.txt의 ETF들을 장중 2시간마다 확인해서
- 지금까지 거래량이 "최근 20거래일의 같은 시각까지 평균 거래량"보다 크게 많은 ETF를
- 등락률·종가 위치·자금 방향과 함께 디스코드로 보냅니다.
같은 시각끼리 비교하므로 장 초반에도 정확합니다 (장 초반 거래량이 원래 많은 것을 감안).
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 ─────────────
RVOL_MIN = float(os.environ.get("RVOL_MIN", "2.0"))            # 평소 같은 시각 대비 이 배수 이상이면 알림
LEVELS = [float(x) for x in os.environ.get("LEVELS", "2,3,5").split(",")]  # 같은 날 이 단계를 새로 넘으면 다시 알림
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "20"))     # 평소 거래량 비교 기간(거래일)
MIN_AVG_DOLLAR = float(os.environ.get("MIN_AVG_DOLLAR", "1000000"))  # 평소 같은 시각까지 거래대금이 이보다 적으면 제외 (거래 적은 ETF 잡음 방지)
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 장 시간이 아니어도 실행 (최근 거래일 기준)
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
SESSION_MIN = 390                                              # 정규장 9:30~16:00 = 390분


def load_etfs() -> dict[str, str]:
    out = {}
    for ln in (ROOT / "etfs.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            t, _, desc = ln.partition(",")
            out[t.strip().upper()] = desc.strip()
    return out


def download_5m(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    data = yf.download(tickers, period="1mo", interval="5m", group_by="ticker",
                       auto_adjust=False, prepost=False, threads=True, progress=False)
    if not isinstance(data.columns, pd.MultiIndex):
        data = pd.concat({tickers[0]: data}, axis=1)
    have = set(data.columns.get_level_values(0))
    return {t: data[t].dropna(subset=["Close"]) for t in tickers if t in have}


def analyze(df: pd.DataFrame) -> dict | None:
    """오늘(가장 최근 거래일) 지금까지 거래량 vs 최근 N일의 같은 시각까지 평균"""
    if df.empty:
        return None
    df = df.copy()
    idx = df.index.tz_convert(NY) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(NY)
    df["date"] = idx.date
    df["tod"] = (idx.hour * 60 + idx.minute) - (9 * 60 + 30)       # 개장 후 경과 분
    df = df[(df["tod"] >= 0) & (df["tod"] < SESSION_MIN)]
    days = sorted(df["date"].unique())
    if len(days) < 6:
        return None
    today, past_days = days[-1], days[:-1][-LOOKBACK_DAYS:]
    td = df[df["date"] == today]
    cutoff = int(td["tod"].max()) + 5                               # 지금까지 경과 분
    past = df[df["date"].isin(past_days) & (df["tod"] < cutoff)]
    past_vol = past.groupby("date")["Volume"].sum()
    past_dollar = (past["Volume"] * past["Close"]).groupby(past["date"]).sum()
    avg_vol, avg_dollar = past_vol.mean(), past_dollar.mean()
    if not avg_vol or avg_dollar < MIN_AVG_DOLLAR:
        return None
    vol = float(td["Volume"].sum())
    last = float(td["Close"].iloc[-1])
    prev_close = float(df[df["date"] == days[-2]]["Close"].iloc[-1])
    hi, lo = float(td["High"].max()), float(td["Low"].min())
    pos = (last - lo) / (hi - lo) * 100 if hi > lo else 50.0
    chg = (last / prev_close - 1) * 100
    if chg > 0 and pos >= 60:
        flow = "유입"
    elif chg < 0 and pos <= 40:
        flow = "이탈"
    else:
        flow = "혼조"
    return {"date": str(today), "elapsed": min(cutoff, SESSION_MIN), "rvol": vol / avg_vol,
            "vol": vol, "dollar": vol * last, "last": last, "chg": chg, "pos": pos, "flow": flow}


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def money(x: float) -> str:
    return f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M"


def width(s: str) -> int:
    """디스코드 고정폭 글꼴에서 한글은 두 칸을 차지"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    """표시 너비 w칸에 맞춰 자르고 공백으로 채움"""
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def send_table(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("설명", 16) + fit("RVOL", 6, True) + fit("등락%", 8, True)
            + fit("위치%", 7, True) + fit("거래대금", 10, True))
    lines = [fit(r["t"], 7) + fit(r["desc"], 16) + f"{r['rvol']:>6.1f}{r['chg']:>+8.2f}{r['pos']:>7.0f}"
             f"{money(r['dollar']):>10}" for r in rows]
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else ""
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first = False


def main():
    now = dt.datetime.now(NY)
    is_open = now.weekday() < 5 and dt.time(9, 35) <= now.time() <= dt.time(17, 30)
    if not is_open and not FORCE:
        print(f"장 시간이 아님 ({now:%Y-%m-%d %H:%M} ET). 종료.")
        return

    etfs = load_etfs()
    data = download_5m(list(etfs))
    results = {t: analyze(df) for t, df in data.items()}
    results = {t: r for t, r in results.items() if r}
    missing = [t for t in etfs if t not in data]
    if not results:
        post("⚠️ ETF 거래량 감시: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    session_date = max(r["date"] for r in results.values())
    elapsed = max(r["elapsed"] for r in results.values())

    # 같은 날 같은 ETF는 RVOL이 다음 단계(2→3→5배)를 넘을 때만 다시 알림
    sent_file = STATE / f"sent_{session_date}.json"
    sent = json.loads(sent_file.read_text()) if sent_file.exists() and not FORCE else {}
    for old in STATE.glob("sent_*.json"):
        if old != sent_file:
            old.unlink()

    hits = []
    for t, r in results.items():
        if r["date"] != session_date or r["rvol"] < RVOL_MIN:
            continue
        level = max([lv for lv in LEVELS if r["rvol"] >= lv], default=RVOL_MIN)
        if sent.get(t, 0) >= level:
            continue
        sent[t] = level
        hits.append({"t": t, "desc": etfs[t], **r})

    pct = elapsed / SESSION_MIN * 100
    stage = "장 마감 기준" if elapsed >= SESSION_MIN else f"개장 후 {elapsed // 60}시간 {elapsed % 60}분 경과 ({pct:.0f}%)"
    if hits:
        hits.sort(key=lambda r: -r["rvol"])
        groups = [("유입", "🟢 유입 (상승 + 당일 고가권)"), ("이탈", "🔴 이탈 (하락 + 당일 저가권)"), ("혼조", "⚪ 혼조 (그 외)")]
        counts = " · ".join(f"{k} {sum(h['flow'] == k for h in hits)}" for k, _ in groups)
        post(f"**ETF 특이 거래량 | {session_date} {stage}**\n"
             f"평소 같은 시각 대비 {RVOL_MIN:g}배 이상 {len(hits)}개 ({counts})")
        for key, title in groups:
            rows = [h for h in hits if h["flow"] == key]
            if rows:
                send_table(title, rows)
        post("RVOL = 지금까지 거래량 ÷ 최근 20거래일 같은 시각까지 평균 · 위치% = 당일 저가(0)~고가(100) 중 현재가 위치")
    elif FORCE:
        post(f"ETF 특이 거래량 | {session_date} {stage}: {RVOL_MIN:g}배 이상 없음 (확인 {len(results)}개)")
    if missing:
        post("⚠️ 시세를 받지 못한 ETF: " + ", ".join(missing))
    if not FORCE:
        sent_file.write_text(json.dumps(sent))
    print(f"완료: 확인 {len(results)}개, 알림 {len(hits)}개")


if __name__ == "__main__":
    sys.exit(main())
