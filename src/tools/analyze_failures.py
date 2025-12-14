from src.core.backtest_analytics import BacktestAnalytics


def analyze_failures(run_id: str) -> None:
    analytics = BacktestAnalytics()

    print(f"Analyzing failures for run: {run_id}")

    # Get decisions
    df = analytics.get_decisions(run_id)

    if df.empty:
        print("No decisions found.")
        return

    print(f"Total decisions: {len(df)}")
    print(f"Actions:\n{df['action'].value_counts()}")

    # Filter for CANDIDATE_IGNORED
    ignored = df[df["action"] == "CANDIDATE_IGNORED"].copy()

    if ignored.empty:
        print("No CANDIDATE_IGNORED rows found.")
        return

    print(f"\nAnalyzing {len(ignored)} ignored candidates (touched_band=1)...")

    # Check failure reasons
    # 1. Regime (Must be Ranging)
    # Note: Strategy checks ADX < Threshold. 'is_ranging' flag should reflect this.
    failed_regime = ignored[ignored["is_ranging"] == 0]
    print(f"Failed Regime (Not Ranging): {len(failed_regime)} ({len(failed_regime) / len(ignored):.1%})")

    # 2. Trend (Must be Uptrend for Long)
    # Strategy checks is_uptrend_1h for Longs.
    # But wait, MeanReversionStrategy checks BOTH Long and Short.
    # If touched_band is True, it could be Upper or Lower.
    # If Lower touched -> Check Uptrend.
    # If Upper touched -> Check Downtrend.

    # Let's infer intended side from band touch
    # If Low <= BB_Lower -> Intended Long
    # If High >= BB_Upper -> Intended Short

    # We need to check if 'touched_band' means ANY touch.
    # In Strategy:
    # if (current['low'] <= (current['bb_lower'] + atr_buffer)) ... touched_band = True
    # elif (current['high'] >= (current['bb_upper'] - atr_buffer)) ... touched_band = True

    # So we need to split by side.
    # But 'side' in decision log might be null if no signal was generated.
    # We can infer from price vs bands.

    # Let's assume we are looking for LONGS (since we care about buying).
    # Filter for potential Longs: Low <= BB_Lower (approx)
    # Or just check if 'is_uptrend_1h' was required but missing.

    # Actually, let's look at the flags directly.

    # 3. RSI
    # Long: RSI < 40
    # Short: RSI > 60
    # We can check how many had RSI in "neutral" zone (40-60).
    failed_rsi = ignored[(ignored["rsi"] >= 40) & (ignored["rsi"] <= 60)]
    print(f"Failed RSI (Neutral 40-60): {len(failed_rsi)} ({len(failed_rsi) / len(ignored):.1%})")

    # 4. Confirmation
    failed_confirm = ignored[ignored["confirmed_1m"] == 0]
    print(f"Failed 1m Confirmation: {len(failed_confirm)} ({len(failed_confirm) / len(ignored):.1%})")

    # 5. Green Candle (Anti-Knife)
    # is_green == 1 required for Long
    # is_red == 1 required for Short
    # Hard to separate without knowing intended side.

    # Let's try to isolate Potential Longs
    # Low <= BB_Lower + Buffer (Buffer is 0.0 now)
    pot_longs = ignored[ignored["low"] <= ignored["bb_lower"]]
    print(f"\nPotential Longs (Low <= BB_Lower): {len(pot_longs)}")

    if not pot_longs.empty:
        pl = pot_longs
        print(
            f"  Failed Uptrend (is_uptrend_1h=0): {len(pl[pl['is_uptrend_1h'] == 0])} "
            f"({len(pl[pl['is_uptrend_1h'] == 0]) / len(pl):.1%})"
        )
        print(f"  Failed RSI (RSI >= 40): {len(pl[pl['rsi'] >= 40])} ({len(pl[pl['rsi'] >= 40]) / len(pl):.1%})")
        print(
            f"  Failed Green Candle (is_green=0): {len(pl[pl['is_green'] == 0])} "
            f"({len(pl[pl['is_green'] == 0]) / len(pl):.1%})"
        )
        print(
            f"  Failed RSI Hook (rsi_hook=0): {len(pl[pl['rsi_hook'] == 0])} "
            f"({len(pl[pl['rsi_hook'] == 0]) / len(pl):.1%})"
        )
        print(
            f"  Failed Confirmation (confirmed_1m=0): {len(pl[pl['confirmed_1m'] == 0])} "
            f"({len(pl[pl['confirmed_1m'] == 0]) / len(pl):.1%})"
        )

        # Intersection: How many passed ALL filters except Confirmation?
        passed_all_but_confirm = pl[
            (pl["is_uptrend_1h"] == 1)
            & (pl["rsi"] < 40)
            & (pl["is_green"] == 1)
            & (pl["rsi_hook"] == 1)
            & (pl["is_ranging"] == 1)
        ]
        print(f"\n  Passed All Filters EXCEPT Confirmation: {len(passed_all_but_confirm)}")
        if not passed_all_but_confirm.empty:
            print(
                f"  Of these, Failed Confirmation: "
                f"{len(passed_all_but_confirm[passed_all_but_confirm['confirmed_1m'] == 0])}"
            )


if __name__ == "__main__":
    # Find latest run
    analytics = BacktestAnalytics()
    runs = analytics.list_runs()
    if runs.empty:
        print("No runs found.")
    else:
        latest_run = runs.iloc[0]["run_id"]
        analyze_failures(latest_run)
