"""
ギャップ・欠損バー診断

data/USDJPY_*.parquet を読み込み、以下を報告する:
  - タイムゾーン・期間・バー数
  - OHLCV 整合チェック (high>=all, low<=all, NaN)
  - ギャップ一覧: 週末ギャップ(正常) / 取引時間内ギャップ(要確認)
  - 土曜バーの有無 (D 以下)
  - close の基本統計

週末ギャップの判定: ギャップ終端バーの曜日が月曜 (day_of_week==0)
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 期待インターバル (秒)
EXPECTED_SEC: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900,
    "H1": 3_600, "H4": 14_400, "D": 86_400, "W": 604_800,
}
GAP_FACTOR = 2   # 期待インターバルの何倍超をギャップとみなすか


def analyze(tf: str) -> None:
    path = DATA_DIR / f"USDJPY_{tf}.parquet"
    if not path.exists():
        print(f"\n{tf}: ファイル未発見 → スキップ")
        return

    df  = pd.read_parquet(path)
    idx = df.index
    n   = len(df)
    tz  = str(getattr(idx, "tz", "None"))

    print(f"\n{'─'*62}")
    print(f" {tf}   n={n:,}   tz={tz}")
    print(f" 期間 : {idx[0]}  →  {idx[-1]}")

    # ── OHLCV 整合チェック ─────────────────────────────────
    nan_n = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    hl_ok = bool((df["high"] >= df[["open", "close", "low"]].max(axis=1)).all())
    ll_ok = bool((df["low"]  <= df[["open", "close", "high"]].min(axis=1)).all())
    print(f" NaN bars={nan_n}   high>=all={hl_ok}   low<=all={ll_ok}")

    # ── ギャップ分析 ───────────────────────────────────────
    exp_td = pd.Timedelta(seconds=EXPECTED_SEC[tf])
    gap_th = exp_td * GAP_FACTOR

    diffs = idx.to_series().diff().dropna()
    gaps  = diffs[diffs > gap_th]

    if len(gaps) == 0:
        print(f" ギャップ(>{int(gap_th.total_seconds()/60):.0f}分): なし")
    else:
        # ギャップ終端が月曜 → 週末を跨いでいる (正常)
        is_weekend = gaps.index.day_of_week == 0
        weekend_g  = gaps[is_weekend]
        trading_g  = gaps[~is_weekend]

        print(f" ギャップ計 {len(gaps)} 件"
              f"  [週末(正常)={len(weekend_g)}  取引時間内={len(trading_g)}]")

        if len(trading_g) > 0:
            top = trading_g.sort_values(ascending=False).head(10)
            print(" ┌─ 取引時間内 ギャップ Top10 ─")
            for ts, td in top.items():
                wday = ["月", "火", "水", "木", "金", "土", "日"][ts.day_of_week]
                print(f" │  {ts}({wday})  gap={td}")
            print(" └───────────────────────────")

        # 週末ギャップの統計 (参考)
        if len(weekend_g) > 0:
            print(f" 週末ギャップ統計: min={weekend_g.min()}  max={weekend_g.max()}"
                  f"  median={weekend_g.median()}")

    # ── 土曜バーチェック (D以下) ───────────────────────────
    if tf not in ("W",):
        sat_n = int((idx.day_of_week == 5).sum())
        if sat_n:
            print(f" !! 土曜バー検出: {sat_n} 件 ← 要調査")
        else:
            print(f" 土曜バー: 0 件 (正常)")

    # ── close 基本統計 ─────────────────────────────────────
    c = df["close"]
    print(f" close: min={c.min():.3f}  max={c.max():.3f}"
          f"  mean={c.mean():.3f}  std={c.std():.3f}")


def main() -> None:
    print("=" * 62)
    print("USDJPY parquet ギャップ / 品質 診断")
    print("=" * 62)
    for tf in ["M1", "M5", "M15", "H1", "H4", "D", "W"]:
        analyze(tf)
    print(f"\n{'='*62}")
    print("診断完了")


if __name__ == "__main__":
    main()
