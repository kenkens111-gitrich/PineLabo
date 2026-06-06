"""
Physics Matrix — Pine→Python 翻訳 + 軸1分析 雛形 (スモークテスト)

目的:
  1. Pine の物理モデル(Vel/Acc/Jerk/Snap 正規化)を Python に正しく翻訳できるか
  2. 特徴量化 → 大move ラベル → lift テーブル → 浅い決定木 が通るか
  3. no-empty-result 原則(常にフルランキング + ベースレート併記)を満たすか

注意:
  実データ(USDJPY)はこの環境から取得不可のため、合成データで配線確認する。
  本番は Claude Code で実ヒストリカルを読み込んで同じ関数を回す。
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

RNG = np.random.default_rng(7)

# =============================================================================
# 1. Pine 翻訳: WMA / HMA / 物理微分 / 正規化
# =============================================================================
def wma(s: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    w = np.arange(1, length + 1, dtype=float)  # Pine WMA: 線形重み 1..n
    return s.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma(s: pd.Series, length: int) -> pd.Series:
    # Pine: ta.wma(2*ta.wma(src,int(len/2)) - ta.wma(src,len), round(sqrt(len)))
    half = int(length / 2)            # int() は切り捨て
    sqrt_len = int(round(np.sqrt(length)))
    return wma(2 * wma(s, half) - wma(s, length), sqrt_len)

def f_norm(s: pd.Series, length: int) -> pd.Series:
    dev = s.rolling(length).std(ddof=0)  # Pine ta.stdev は母標準偏差(ddof=0)
    return np.where(dev.values != 0, s.values / dev.values, 0.0)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()  # Pine RMA 相当

def compute_physics(df: pd.DataFrame, len_smooth=6, len_hma=72, len_norm=100) -> pd.DataFrame:
    src_smooth = hma(df["close"], len_smooth)
    mid_line = hma(src_smooth, len_hma)
    raw_vel = mid_line.diff()
    raw_acc = raw_vel.diff()
    raw_jerk = raw_acc.diff()
    raw_snap = raw_jerk.diff()
    out = pd.DataFrame(index=df.index)
    out["n_vel"] = f_norm(raw_vel, len_norm)
    out["n_acc"] = f_norm(raw_acc, len_norm)
    out["n_jerk"] = f_norm(raw_jerk, len_norm)
    out["n_snap"] = f_norm(raw_snap, len_norm)
    return out

# =============================================================================
# 2. 特徴量化 (エントリー時点で過去データのみ = リークなし)
# =============================================================================
def make_features(phys: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=phys.index)
    for layer in ["n_vel", "n_acc", "n_jerk", "n_snap"]:
        nm = layer.replace("n_", "")
        f[f"{nm}_slope_up"] = phys[layer] > phys[layer].shift(1)   # 傾き上昇
        f[f"{nm}_pol_pos"] = phys[layer] > 0                       # 極性 正
    # 階層関係(位置関係の核)
    f["hier_acc_gt_vel"] = phys["n_acc"] > phys["n_vel"]
    f["hier_jerk_gt_acc"] = phys["n_jerk"] > phys["n_acc"]
    f["hier_snap_gt_jerk"] = phys["n_snap"] > phys["n_jerk"]
    # Vel パニック領域(force閾値 2.0)
    f["vel_strong_up"] = phys["n_vel"] >= 2.0
    return f

# =============================================================================
# 3. 大move ラベル (今後 H 本で 閾値*ATR 超)
# =============================================================================
def label_big_move(df: pd.DataFrame, atr_s: pd.Series, H=20, thr_atr=1.5) -> pd.Series:
    fwd_max = df["high"].shift(-1).rolling(H).max().shift(-(H - 1))
    up_move = (fwd_max - df["close"]) / atr_s
    return (up_move >= thr_atr).astype("Int8")  # 1=大上昇が来た

# =============================================================================
# 4. 軸1分析: 条件付き発生率 + lift + 浅い決定木 (no-empty-result)
# =============================================================================
def lift_table(feat: pd.DataFrame, label: pd.Series, min_n=30) -> pd.DataFrame:
    base = label.mean()  # ベースレート(平時の大move発生率)
    rows = []
    for col in feat.columns:
        for state in [True, False]:
            mask = feat[col] == state
            n = int(mask.sum())
            rate = label[mask].mean() if n > 0 else np.nan
            rows.append({
                "feature": col, "state": state, "n": n,
                "rate": rate, "baseline": base,
                "lift": (rate / base) if (base and n > 0) else np.nan,
                "enough_n": n >= min_n,  # 足切りは「消す」でなく「印」
            })
    t = pd.DataFrame(rows).sort_values("lift", ascending=False, na_position="last")
    return t.reset_index(drop=True)

def tree_rules(feat: pd.DataFrame, label: pd.Series, max_depth=3, min_leaf=50):
    X = feat.astype(int).values
    y = label.astype(int).values
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                 criterion="entropy", random_state=0)
    clf.fit(X, y)
    return export_text(clf, feature_names=list(feat.columns))

# =============================================================================
# 5. 合成データ (momentum バーストを埋め込み、大move が存在する系列)
# =============================================================================
def synth_prices(n=30000):
    drift = np.zeros(n)
    i = 0
    while i < n:
        run = RNG.integers(30, 200)
        d = RNG.normal(0, 0.0008) * (RNG.random() < 0.35)  # 時々トレンド注入
        drift[i:i + run] = d
        i += run
    ret = drift + RNG.normal(0, 0.0006, n)
    close = 150 * np.exp(np.cumsum(ret))
    spread = np.abs(RNG.normal(0, 0.0004, n)) * close
    high = close + spread * RNG.random(n)
    low = close - spread * RNG.random(n)
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

# =============================================================================
# 実行
# =============================================================================
if __name__ == "__main__":
    df = synth_prices()
    phys = compute_physics(df)
    atr_s = atr(df, 14)
    feat = make_features(phys)
    label = label_big_move(df, atr_s, H=20, thr_atr=1.5)

    valid = phys.notna().all(axis=1) & atr_s.notna() & label.notna()
    feat_v, label_v = feat[valid], label[valid].astype(int)

    print("=" * 70)
    print("[1] Pine→Python 翻訳チェック")
    print("=" * 70)
    print(phys[valid].describe().round(3).T[["mean", "std", "min", "max"]])
    print(f"\n有効バー数: {valid.sum()} / {len(df)}  (HMA/正規化のウォームアップ分を除外)")
    print(f"大move(今後20本で1.5ATR超上昇) 発生率 = {label_v.mean():.3%}  ← これがベースライン")

    print("\n" + "=" * 70)
    print("[2] 軸1: 条件付き発生率テーブル (lift順・全件表示・足切りは印のみ)")
    print("=" * 70)
    t = lift_table(feat_v, label_v, min_n=30)
    show = t.copy()
    show["rate"] = (show["rate"] * 100).round(1)
    show["baseline"] = (show["baseline"] * 100).round(1)
    show["lift"] = show["lift"].round(2)
    print(show.to_string(index=False))

    print("\n" + "=" * 70)
    print("[3] 軸1: 浅い決定木が抽出した読めるルール (max_depth=3)")
    print("=" * 70)
    print(tree_rules(feat_v, label_v))

    print("=" * 70)
    print("[no-empty-result 確認] 上記テーブルは基準未達も含め必ず全件 + baseline 併記済み")
    print("=" * 70)
