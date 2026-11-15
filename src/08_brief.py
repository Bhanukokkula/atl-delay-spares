"""Step 8: build the 2-page brief (outputs/brief.pdf).

Every number in the brief is read from the CSV files written by steps 2-7,
so nothing is typed in by hand.

Reads:  outputs/*.csv, outputs/spares_by_s.png
Writes: outputs/brief.pdf
"""
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from pypdf import PdfReader

import config

OUT = "outputs"
DEMO_URL = "https://bhanukokkula.github.io/atl-delay-spares/"
REPO_URL = "https://github.com/Bhanukokkula/atl-delay-spares"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    phase1 = pd.read_csv(f"{OUT}/phase1_summary.csv")
    results = pd.read_csv(f"{OUT}/phase3_results.csv")
    spares = pd.read_csv(f"{OUT}/phase4_summary.csv")
    sens = pd.read_csv(f"{OUT}/phase4_sensitivity.csv")
    drift = pd.read_csv(f"{OUT}/phase7a_drift_summary.csv").set_index("version")
    boot = pd.read_csv(f"{OUT}/phase7c_bootstrap.csv").set_index("metric")
    drift_months = pd.read_csv(f"{OUT}/phase7a_drift_offset.csv")

    # ---- numbers ----
    n_departures = int(phase1["departures"].sum())
    n_cancelled = int(phase1["cancelled"].sum())
    first_month = phase1["month"].iloc[0]
    last_month = phase1["month"].iloc[-1]

    def result(model, split, col):
        row = results[(results["model"] == model) & (results["split"] == split)]
        return row[col].iloc[0]

    chosen = config.CHOSEN_NAME
    test_rate = result(chosen, "test", "actual_delay_rate")
    test_flights = int(result(chosen, "test", "flights"))
    auc = result(chosen, "test", "roc_auc")
    pr = result(chosen, "test", "pr_auc")
    cal = result(chosen, "test", "calibration_error")
    mean_pred = result(chosen, "test", "mean_predicted")
    auc_a = result("lgbm_all", "test", "roc_auc")
    auc_b = result("lgbm_swap_safe", "test", "roc_auc")
    auc_c = result("lgbm_no_inbound", "test", "roc_auc")
    auc_bw = result("lgbm_swap_safe_weather", "test", "roc_auc")
    auc_lr = result("logistic_all", "test", "roc_auc")
    val_b = result("lgbm_swap_safe", "valid", "roc_auc")
    val_bw = result("lgbm_swap_safe_weather", "valid", "roc_auc")

    def ci(metric, digits):
        return f"{boot.loc[metric, 'ci95_low']:.{digits}f}–{boot.loc[metric, 'ci95_high']:.{digits}f}"

    def spare(method, col):
        return spares.loc[spares["method"] == method, col].iloc[0]

    opt_min = spare("optimization", "expected_min_removed_per_day")
    greedy_min = spare("greedy", "expected_min_removed_per_day")
    random_min = spare("random", "expected_min_removed_per_day")
    greedy_pct = spare("greedy", "pct_of_optimization")
    random_pct = spare("random", "pct_of_optimization")
    opt_actual = spare("optimization", "actual_min_removed_per_day_if_R_holds")
    greedy_actual = spare("greedy", "actual_min_removed_per_day_if_R_holds")
    opt_hit = spare("optimization", "share_assigned_really_delayed_pct")
    greedy_hit = spare("greedy", "share_assigned_really_delayed_pct")
    random_hit = spare("random", "share_assigned_really_delayed_pct")
    opt_n = spare("optimization", "assignments_per_day")
    greedy_n = spare("greedy", "assignments_per_day")
    actual_gap_pct = 100 * (opt_actual - greedy_actual) / opt_actual

    def greedy_share(s, r, h):
        row = sens[(sens["S"] == s) & (sens["R"] == r) & (sens["H"] == h) & (sens["method"] == "greedy")]
        return row["pct_of_optimization"].iloc[0]

    greedy_s_values = []
    for s in sorted(sens["S"].unique()):
        greedy_s_values.append(greedy_share(s, 0.6, 4))
    greedy_h2 = greedy_share(3, 0.6, 2)
    greedy_h8 = greedy_share(3, 0.6, 8)

    n_months = len(drift_months)
    n_worse = int((drift_months["calibration_error_after"] > drift_months["calibration_error_before"]).sum())

    # ---- styles ----
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.2, leading=12, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=15, leading=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=5, spaceAfter=3)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2)
    finding = ParagraphStyle("finding", parent=body, leftIndent=12)
    header = ParagraphStyle("header", parent=body, fontName="Helvetica-Bold", fontSize=8.5, leading=10, spaceAfter=0)

    story = []
    story.append(Paragraph("Delay risk and spare aircraft placement for DL departures from ATL", h1))
    story.append(Paragraph(
        "Independent portfolio project on public data (U.S. DOT BTS flights, Iowa Environmental Mesonet "
        "weather). Not affiliated with or endorsed by Delta Air Lines. Uses no Delta internal data. "
        f"Interactive demo: <a href='{DEMO_URL}' color='#1c5cab'>{DEMO_URL}</a>", small))

    story.append(Paragraph("Business question", h2))
    story.append(Paragraph(
        "Two hours before each scheduled departure, which Delta mainline flights leaving Atlanta are at "
        "risk of a 15+ minute departure delay? And if an operation has a few spare aircraft, where should "
        "they go, and does an optimization model beat a simple rule?", body))
    story.append(Paragraph(
        f"Data: BTS On-Time Performance, {first_month} to {last_month}, reporting carrier DL only "
        f"({n_departures:,} scheduled ATL departures, of which {n_cancelled:,} cancelled flights are excluded), "
        f"plus ATL weather observations. Test month {last_month}: {test_flights:,} flights, "
        f"{pct(test_rate)} delayed 15+ minutes. Ranges in brackets are 95% intervals from resampling test days.", body))

    story.append(Image(f"{OUT}/spares_by_s.png", width=5.1 * inch, height=3.4 * inch))
    story.append(Paragraph(
        "Expected delay minutes removed per day in the test month, by number of spare aircraft. "
        "A spare removes 60% of a flight's risk minutes and is then busy for 4 hours.", small))

    story.append(Paragraph("Three findings", h2))
    story.append(Paragraph(
        f"<b>1. The model ranks risk usefully, using only information known 2 hours out.</b> "
        f"Test ROC-AUC {auc:.3f} [{ci('test_roc_auc', 3)}], PR-AUC {pr:.3f} (random guessing: 0.5 and "
        f"{test_rate:.3f}). Flights the greedy rule gives a spare were really delayed {greedy_hit:.1f}% of the time "
        f"vs {random_hit:.1f}% for random picks: {boot.loc['hit_rate_greedy_minus_random_pts', 'estimate']:.1f} "
        f"points better [{ci('hit_rate_greedy_minus_random_pts', 1)}].", finding))
    story.append(Paragraph(
        f"<b>2. Greedy is close to the optimization model.</b> Greedy gets {greedy_pct:.1f}% "
        f"[{ci('greedy_pct_of_optimal', 1)}%] of the optimal expected minutes ({greedy_min:.1f} vs {opt_min:.1f} per "
        f"day with 3 spares). On what actually happened, the gap is {actual_gap_pct:.1f}% "
        f"({greedy_actual:.1f} vs {opt_actual:.1f} minutes per day).", finding))
    story.append(Paragraph(
        f"<b>3. A daily correction fixes most calibration drift.</b> Without it, the model under-predicts bad "
        f"months and over-predicts calm ones (18-month backtest, correlation "
        f"{drift.loc['before', 'month_severity_correlation']:.2f}). Shifting each day's predictions to match the "
        f"previous 7 days' delay rate cut mean calibration error from {drift.loc['before', 'mean_calibration_error']:.4f} "
        f"to {drift.loc['after', 'mean_calibration_error']:.4f} and the correlation to "
        f"{drift.loc['after', 'month_severity_correlation']:.2f}, without changing ranking (mean ROC-AUC "
        f"{drift.loc['before', 'mean_roc_auc']:.4f} vs {drift.loc['after', 'mean_roc_auc']:.4f}).", finding))

    story.append(PageBreak())

    story.append(Paragraph("Decision results (test month, 3 spares, R = 0.6, H = 4 hours)", h2))
    table = [
        [Paragraph("Method", header), Paragraph("Expected minutes removed per day", header),
         Paragraph("% of optimal", header), Paragraph("Spare uses per day", header),
         Paragraph("Assigned flights really delayed 15+", header)],
        ["Optimization (PuLP)", f"{opt_min:.1f}", "100.0%", f"{opt_n:.1f}", f"{opt_hit:.1f}%"],
        ["Greedy (highest risk first)", f"{greedy_min:.1f}", f"{greedy_pct:.1f}%", f"{greedy_n:.1f}", f"{greedy_hit:.1f}%"],
        ["Random (20 runs)", f"{random_min:.1f}", f"{random_pct:.1f}%",
         f"{spare('random', 'assignments_per_day'):.1f}", f"{random_hit:.1f}%"],
    ]
    t = Table(table, colWidths=[1.9 * inch, 1.4 * inch, 0.9 * inch, 1.0 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"The optimization wins by fitting more assignments into the day ({opt_n:.1f} vs {greedy_n:.1f}), "
        f"not by finding riskier flights: greedy's first pick can block a spare for 4 hours. Greedy stays at "
        f"{min(greedy_s_values):.1f}-{max(greedy_s_values):.1f}% of optimal for 1 to 8 spares. The gap grows when "
        f"spares turn around faster (greedy {greedy_h2:.1f}% at H = 2 hours) and shrinks when they are tied up "
        f"longer ({greedy_h8:.1f}% at H = 8 hours). The optimization would matter more with short spare cycles, "
        f"several airports or fleet types competing for the same spares, or crew and gate limits that make "
        f"assignments interact. R only rescales the minutes; it does not change which flights are chosen.", body))

    story.append(Paragraph("Model range and what weather added", h2))
    story.append(Paragraph(
        f"BTS records the tail number that actually flew, so aircraft swaps leak into inbound-aircraft features. "
        f"Test ROC-AUC: all features {auc_a:.3f}; swap-safe features {auc_b:.3f}; no inbound features {auc_c:.3f}; "
        f"logistic regression {auc_lr:.3f}. Adding ATL weather (thunderstorm, precipitation, visibility, ceiling, "
        f"gust from the last observation before decision time) raised validation ROC-AUC from {val_b:.3f} to "
        f"{val_bw:.3f}, so it was chosen, and test ROC-AUC from {auc_b:.3f} to {auc_bw:.3f}. But the test gain "
        f"of {boot.loc['test_roc_auc_gain_weather_vs_b', 'estimate']:.4f} has a 95% interval of "
        f"{ci('test_roc_auc_gain_weather_vs_b', 4)}, which includes zero: weather may not really help. "
        f"Decisions use swap-safe + weather with the daily correction (test calibration error {cal:.3f}).", body))

    story.append(Paragraph("Limitations", h2))
    limits = [
        f"The daily correction only fixes the overall level. On the test month the corrected model predicts "
        f"{pct(mean_pred)} vs {pct(test_rate)} actual; in the backtest {n_worse} of {n_months} months got slightly worse calibration.",
        "Tail numbers are what actually flew, not the plan made 2 hours earlier. Swaps affect all inbound features, including the one kept.",
        "R (share of delay a spare removes) and H (hours a spare is busy) are assumptions, not measured. Spares are assumed to be at ATL, interchangeable, and crewed.",
        "Risk minutes use the average delay of delayed flights by departure hour from training data, not a per-flight forecast of minutes.",
        "Cancellations are excluded because BTS does not record when they were announced. Delta Connection flights are reported by other carriers and are out of scope.",
        "Weather is ATL observations only: no forecasts, no weather at destinations or along routes. No gate, crew, or passenger-connection data. One airport, one test month.",
    ]
    for item in limits:
        story.append(Paragraph(item, bullet, bulletText="•"))

    story.append(Paragraph("What I would do with Delta's internal data", h2))
    internal = [
        "Use the planned tail assignment at decision time, which removes the swap problem and lets turn time back in.",
        "Measure R and H from real spare and swap records instead of assuming them.",
        "Add crew connections, gate availability, and inbound flights from across the network, not just into ATL.",
        "Add weather forecasts and air-traffic programs (ground stops, delay programs) known at decision time.",
        "Predict delay minutes directly, not just the chance of a 15+ minute delay.",
    ]
    for item in internal:
        story.append(Paragraph(item, bullet, bulletText="•"))
    story.append(Paragraph(f"Code and method: <a href='{REPO_URL}' color='#1c5cab'>{REPO_URL}</a>", small))

    doc = SimpleDocTemplate(f"{OUT}/brief.pdf", pagesize=letter,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title="DL ATL delay risk and spare aircraft brief")
    doc.build(story)

    pages = len(PdfReader(f"{OUT}/brief.pdf").pages)
    print(f"Wrote {OUT}/brief.pdf ({pages} pages)")
    if pages > 2:
        raise RuntimeError("Brief is longer than 2 pages")


if __name__ == "__main__":
    main()
