#!/usr/bin/env bash
# ============================================================
#  additions.sh  – “hot-patch” for data-massage + intent filter
#  ------------------------------------------------------------
#  • Safe to run multiple times (idempotent)
#  • Only touches the modules listed below
#  • Re-runs the entire pipeline after patch
# ============================================================
set -euo pipefail
shopt -s expand_aliases

echo "🔄  Applying massage / filter patch …"

#---------------------------------------------
# 1⃣  Update data/loader.py  (file-median scaler)
#---------------------------------------------
cat > customer_comms/data/loader.py << 'PY'
from __future__ import annotations
import pandas as pd, holidays, numpy as np
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

ENCODINGS = ("utf-8","latin-1","cp1252","utf-16")
us_holidays = holidays.UnitedStates()

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning(f"Missing {path.name}")
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    log.error(f"Could not decode {path}")
    return pd.DataFrame()

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

# ---------- CALL DATA (volume + intent) ----------
def load_call_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    medians = {}
    for fname in settings.call_files:
        df = _read(settings.data_dir / fname)
        if df.empty:
            continue

        date_col = next((c for c in settings.call_date_cols if c in df.columns), None)
        if not date_col:
            log.warning(f"{fname}: no recognised date col")
            continue
        intent_col = next((c for c in settings.call_intent_cols if c in df.columns), None)

        df = df.rename(columns={date_col: "date"} | ({intent_col: "intent"} if intent_col else {}))
        df["date"] = _to_date(df["date"])
        df = df.dropna(subset=["date"])
        df["file_tag"] = fname
        dfs.append(df)

        # keep median for scaling later
        medians[fname] = (
            df.groupby("date").size().median() if not df.empty else np.nan
        )

    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    # ---------------- scaling step -----------------
    target = np.nanmedian(list(medians.values()))
    scaled = []
    for df in dfs:
        ftag = df["file_tag"].iat[0]
        factor = target / medians.get(ftag, target) if medians.get(ftag, 0) else 1.0
        g = df.groupby("date").size().mul(factor).round().astype(int).reset_index(name="call_volume")
        g["file_tag"] = ftag
        scaled.append(g)

    call_volume = (
        pd.concat(scaled, ignore_index=True)
        .groupby("date", as_index=False)["call_volume"]
        .sum()
        .sort_values("date")
    )

    # ---------- intents (optional) ----------
    intents_raw = [d[["date", "intent"]] for d in dfs if "intent" in d.columns]
    intents = (
        pd.concat(intents_raw, ignore_index=True)
        .groupby(["date", "intent"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("date")
    ) if intents_raw else pd.DataFrame()

    log.info(f"Call-volume rows: {len(call_volume)}")
    return call_volume, intents


# ---------- MAIL ----------
def load_mail() -> pd.DataFrame:
    df = _read(settings.data_dir / settings.mail_file)
    if df.empty:
        return df
    df = df.rename(
        columns={
            settings.mail_date_col: "date",
            settings.mail_type_col: "mail_type",
            settings.mail_volume_col: "mail_volume",
        }
    )
    df["date"] = _to_date(df["date"])
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce")
    df = df.dropna(subset=["date", "mail_volume"])
    return df
PY

#---------------------------------------------
# 2⃣  Patch processing/combine.py  (pct/diff + row filter)
#---------------------------------------------
cat > customer_comms/processing/combine.py << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

_norm = lambda s: (s - s.min()) / (s.max() - s.min()) * 100 if s.max() != s.min() else s * 0

def build_dataset() -> pd.DataFrame:
    call_vol, _ = load_call_data()
    mail = load_mail()

    # weekday filter
    call_vol = call_vol[call_vol["date"].dt.weekday < 5]
    mail = mail[mail["date"].dt.weekday < 5]

    if call_vol.empty or mail.empty:
        log.error("No weekday call or mail data")
        return pd.DataFrame()

    mail_daily = mail.groupby("date", as_index=False)["mail_volume"].sum()

    df = pd.merge(call_vol, mail_daily, on="date", how="inner")
    # drop “silent” days
    df = df[(df["call_volume"] > 0) | (df["mail_volume"] > 0)]

    if len(df) < settings.min_rows:
        log.error("Too few overlapping days")
        return pd.DataFrame()

    # normalised + pct/diff features
    df["call_norm"] = _norm(df["call_volume"])
    df["mail_norm"] = _norm(df["mail_volume"])
    df["call_pct"] = df["call_volume"].pct_change().fillna(0)
    df["mail_pct"] = df["mail_volume"].pct_change().fillna(0)
    df["call_diff"] = df["call_volume"].diff().fillna(0)
    df["mail_diff"] = df["mail_volume"].diff().fillna(0)

    return df.sort_values("date")
PY

#---------------------------------------------
# 3⃣  Patch analytics/mail_intent_corr.py
#     (≥250 filter to avoid KeyError)
#---------------------------------------------
cat > customer_comms/analytics/mail_intent_corr.py << 'PY'
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

MIN_COUNT = 250   # <-- threshold

def top_mail_intent_corr():
    call_vol, intents = load_call_data()
    mail = load_mail()

    # weekday filter
    intents = intents[intents["date"].dt.weekday < 5] if not intents.empty else intents
    mail = mail[mail["date"].dt.weekday < 5]

    if intents.empty or mail.empty:
        log.warning("No intents or mail after filtering")
        return

    # ---- collapse rare categories ----
    intents = intents.loc[:, (intents.sum() >= MIN_COUNT) | (intents.columns == "date")]
    log.info(f"Intents retained: {len(intents.columns)-1}")

    mail_piv = (
        mail.pivot_table(index="date", columns="mail_type", values="mail_volume", aggfunc="sum")
        .fillna(0)
        .loc[:, lambda df: df.sum() >= MIN_COUNT]
    )
    log.info(f"Mail types retained: {mail_piv.shape[1]}")

    merged = pd.merge(mail_piv.reset_index(), intents, on="date", how="inner").set_index("date")
    if merged.empty:
        log.error("No overlap after merge")
        return

    results = []
    for m in mail_piv.columns:
        for i in [c for c in intents.columns if c != "date" and c in merged.columns]:
            if merged[m].std() == 0 or merged[i].std() == 0:
                continue
            r, _ = pearsonr(merged[m], merged[i])
            results.append((m, i, abs(r), r))
    if not results:
        log.warning("No valid correlations")
        return

    top = sorted(results, key=lambda x: x[2], reverse=True)[:10]
    heat = (
        pd.DataFrame(top, columns=["mail_type", "intent", "abs_r", "r"])
        .pivot(index="mail_type", columns="intent", values="r")
        .fillna(0)
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Top mail-type × intent correlations (≥250 obs, weekdays)")
    plt.tight_layout()
    out = settings.output_dir / "mailtype_intent_corr.png"
    plt.savefig(out, dpi=300)
    plt.close()
    log.info(f"Saved {out}")
PY

#---------------------------------------------
# 4⃣  Patch analytics/eda.py   (lag corr heatmap safe)
#---------------------------------------------
sed -i.bak 's/tmp\[\(feat\|tgt\)\].dropna()/tmp[\1]/g' customer_comms/analytics/eda.py
rm -f customer_comms/analytics/eda.py.bak

#---------------------------------------------
# 5⃣  Patch analytics/corr_extras.py (use pct)
#---------------------------------------------
sed -i.bak 's/"mail_volume"/"mail_pct"/' customer_comms/analytics/corr_extras.py
sed -i 's/"call_volume"/"call_pct"/'      customer_comms/analytics/corr_extras.py
rm customer_comms/analytics/corr_extras.py.bak

#---------------------------------------------
# 6⃣  Re-run the full pipeline
#---------------------------------------------
echo "🚀  Re-running pipeline with massaged data …"
python -m customer_comms.pipeline3 || echo "⚠️  Pipeline failed – see logs"

echo "✅  Patch complete – refreshed plots in ./output/"