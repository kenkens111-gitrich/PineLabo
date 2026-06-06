"""
USDJPY ヒストリカル取得 — Dukascopy tick → 七足 parquet

tick-vault で Dukascopy から tick をダウンロードし、
  tick → M1 → M5 / M15 / H1 / H4 / D / W
の順で OHLCV を生成して data/USDJPY_{TF}.parquet に保存する。

全インデックスは UTC(GMT) で固定。
D の日替わりは 00:00 UTC、W のラベルは週末金曜 (W-FRI)。
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tick_vault import download_range, read_tick_data
from tick_vault.config import reload_config

# ─────────────────────────────────────────────
SYMBOL    = "USDJPY"
START     = datetime(2025, 1, 1, tzinfo=timezone.utc)
END       = datetime(2026, 1, 1, tzinfo=timezone.utc)

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "tick_vault_cache"   # tick-vault 内部キャッシュ

reload_config(base_directory=str(CACHE_DIR))

# pandas resample ルール。
# W-FRI: closed='right'/label='right' (pandas デフォルト)
# → 期間は (前週金曜, 今週金曜] で土日データ無し = 実質月〜金バー
HIGHER_TF_RULES: dict[str, str] = {
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D":   "1D",
    "W":   "W-FRI",
}
# ─────────────────────────────────────────────


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """tick DataFrame → M1 OHLCV (UTC インデックス)"""
    # time 列が timezone-naive (UTC) → UTC に明示ローカライズ
    times = pd.DatetimeIndex(ticks["time"]).tz_localize("UTC")
    mid   = pd.Series((ticks["ask"].values + ticks["bid"].values) / 2, index=times)
    vol   = pd.Series(
        ticks["ask_volume"].values + ticks["bid_volume"].values,
        index=times,
        dtype=float,
    )
    ohlcv          = mid.resample("1min").ohlc()
    ohlcv["volume"] = vol.resample("1min").sum()
    return ohlcv.dropna(subset=["open"])


def resample_ohlcv(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """M1 OHLCV → 上位足 OHLCV (空バーは除去)"""
    return (
        m1.resample(rule)
        .agg(
            open=("open",   "first"),
            high=("high",   "max"),
            low=("low",    "min"),
            close=("close",  "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )


async def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    print(f"[1] {SYMBOL} tick ダウンロード  {START.date()} → {END.date()}")
    await download_range(SYMBOL, START, END)

    print("[2] tick 読み込み")
    ticks = read_tick_data(SYMBOL, START, END, strict=False)
    print(f"    {len(ticks):,} ticks  columns={list(ticks.columns)}")

    print("[3] M1 リサンプリング")
    m1   = ticks_to_m1(ticks)
    path = DATA_DIR / "USDJPY_M1.parquet"
    m1.to_parquet(path)
    print(f"    M1    {len(m1):>8,} バー  → {path.name}"
          f"  ({m1.index[0]} … {m1.index[-1]})")

    print("[4] 上位足リサンプリング")
    for tf, rule in HIGHER_TF_RULES.items():
        ohlcv = resample_ohlcv(m1, rule)
        path  = DATA_DIR / f"USDJPY_{tf}.parquet"
        ohlcv.to_parquet(path)
        print(f"    {tf:<4}  {len(ohlcv):>8,} バー  → {path.name}")

    print("\n完了")


if __name__ == "__main__":
    asyncio.run(main())
