def lag_scan(df, feat="mail_volume", tgt="call_volume"):
    lags = range(0, settings.max_lag + 1)
    vals = []

    for lag in lags:
        s1 = df[feat]
        s2 = df[tgt].shift(-lag)            # look-ahead
        valid = s1.notna() & s2.notna()     # keep rows present in *both*

        if valid.sum() < 2:                 # need ≥ 2 points
            vals.append(float("nan"))
            continue

        r, _ = pearsonr(s1[valid], s2[valid])
        vals.append(r)

    plt.figure(figsize=(10, 4))
    plt.plot(lags, vals, marker="o")
    plt.grid(ls="--", alpha=.4)
    plt.title("Mail → Call cross-correlation (weekdays)")
    plt.xlabel("Lag (days)"); plt.ylabel("r")
    out = settings.output_dir / "lag_scan.png"
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    log.info(f"Saved {out}")