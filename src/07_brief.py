"""Step 7: build the 2-page brief (outputs/brief.pdf).

Every number in the brief is read from the CSV files written by steps 2-6,
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

OUT = "outputs"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    phase1 = pd.read_csv(f"{OUT}/phase1_summary.csv")
    results = pd.read_csv(f"{OUT}/phase3_results.csv")
    spares = pd.read_csv(f"{OUT}/phase4_summary.csv")
    sens = pd.read_csv(f"{OUT}/phase4_sensitivity.csv")
    backtest = pd.read_csv(f"{OUT}/phase5_backtest.csv")

    # ---- numbers ----
    n_departures = int(phase1["departures"].sum())
    n_cancelled = int(phase1["cancelled"].sum())
    first_month = phase1["month"].iloc[0]
    last_month = phase1["month"].iloc[-1]

    def result(model, split, col):
        row = results[(results["model"] == model) & (results["split"] == split)]
        return row[col].iloc[0]

    test_rate = result("lgbm_swap_safe", "test", "actual_delay_rate")
    test_flights = int(result("lgbm_swap_safe", "test", "flights"))
    auc_a = result("lgbm_all", "test", "roc_auc")
    auc_b = result("lgbm_swap_safe", "test", "roc_auc")
    auc_c = result("lgbm_no_inbound", "test", "roc_auc")
    auc_lr = result("logistic_all", "test", "roc_auc")
    pr_b = result("lgbm_swap_safe", "test", "pr_auc")
    cal_b = result("lgbm_swap_safe", "test", "calibration_error")
    under_b = 100 * (result("lgbm_swap_safe", "test", "actual_delay_rate")
                     - result("lgbm_swap_safe", "test", "mean_predicted"))

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

    bt_min = backtest.loc[backtest["roc_auc"].idxmin()]
    bt_max = backtest.loc[backtest["roc_auc"].idxmax()]
    worst_cal = backtest.loc[backtest["predicted_minus_actual_pts"].idxmin()]
    bt_corr = backtest["actual_delay_rate"].corr(backtest["predicted_minus_actual_pts"])

    # ---- styles ----
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=12.5, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=15, leading=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=6, spaceAfter=3)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2)
    finding = ParagraphStyle("finding", parent=body, leftIndent=12)
    header = ParagraphStyle("header", parent=body, fontName="Helvetica-Bold", fontSize=8.5, leading=10, spaceAfter=0)

    story = []
    story.append(Paragraph("Delay risk and spare aircraft placement for DL departures from ATL", h1))
    story.append(Paragraph(
        "Independent portfolio project on public U.S. DOT (BTS) data. Not affiliated with or endorsed "
        "by Delta Air Lines. Uses no Delta internal data.", small))

    story.append(Paragraph("Business question", h2))
    story.append(Paragraph(
        "Two hours before each scheduled departure, which Delta mainline flights leaving Atlanta are at "
        "risk of a 15+ minute departure delay? And if an operation has a few spare aircraft, where should "
        "they go, and does an optimization model beat a simple rule?", body))
    story.append(Paragraph(
        f"Data: BTS On-Time Performance, {first_month} to {last_month}, reporting carrier DL only "
        f"({n_departures:,} scheduled ATL departures, of which {n_cancelled:,} cancelled flights are excluded). "
        f"Test month {last_month}: {test_flights:,} flights, {pct(test_rate)} delayed 15+ minutes.", body))

    story.append(Image(f"{OUT}/spares_by_s.png", width=5.4 * inch, height=3.6 * inch))
    story.append(Paragraph(
        "Expected delay minutes removed per day in the test month, by number of spare aircraft. "
        "A spare removes 60% of a flight's risk minutes and is then busy for 4 hours.", small))

    story.append(Paragraph("Three findings", h2))
    story.append(Paragraph(
        f"<b>1. The model ranks risk usefully, using only information known 2 hours out.</b> "
        f"LightGBM on swap-safe features scores ROC-AUC {auc_b:.3f} and PR-AUC {pr_b:.3f} on the test month "
        f"(random guessing: 0.5 and {test_rate:.3f}). Flights it picks for spares were really delayed "
        f"{greedy_hit:.1f}% of the time (greedy) vs {random_hit:.1f}% for random picks.", finding))
    story.append(Paragraph(
        f"<b>2. Greedy is close to the optimization model.</b> Greedy gets {greedy_pct:.1f}% of the optimal "
        f"expected minutes ({greedy_min:.1f} vs {opt_min:.1f} per day with 3 spares). Measured on what actually "
        f"happened, the gap is {actual_gap_pct:.1f}% ({greedy_actual:.1f} vs {opt_actual:.1f} minutes per day).",
        finding))
    story.append(Paragraph(
        f"<b>3. Calibration drifts with the month's delay level.</b> In an 18-month backtest, ROC-AUC stayed "
        f"between {bt_min['roc_auc']:.3f} ({bt_min['month']}) and {bt_max['roc_auc']:.3f} ({bt_max['month']}), "
        f"but the model under-predicts bad months and over-predicts calm ones (correlation {bt_corr:.2f}); "
        f"worst was {worst_cal['month']} at {worst_cal['predicted_minus_actual_pts']:.1f} points.",
        finding))

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
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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

    story.append(Paragraph("Model range", h2))
    story.append(Paragraph(
        f"BTS records the tail number that actually flew, so aircraft swaps leak into inbound-aircraft features. "
        f"Test ROC-AUC: all features {auc_a:.3f}; swap-safe features {auc_b:.3f} (used for decisions); "
        f"no inbound features {auc_c:.3f}. Logistic regression baseline: {auc_lr:.3f}. "
        f"Test calibration error for the chosen model: {cal_b:.3f}.", body))

    story.append(Paragraph("Limitations", h2))
    limits = [
        f"The chosen model under-predicts the test month's delay rate by {under_b:.1f} points, so risk minutes are slightly understated. Not recalibrated.",
        "Tail numbers are what actually flew, not the plan made 2 hours earlier. Swaps affect all inbound features, including the one kept.",
        "R (share of delay a spare removes) and H (hours a spare is busy) are assumptions, not measured. Spares are assumed to be at ATL, interchangeable, and crewed.",
        "Risk minutes use the average delay of delayed flights by departure hour from training data, not a per-flight forecast of minutes.",
        "Cancellations are excluded because BTS does not record when they were announced. Delta Connection flights are reported by other carriers and are out of scope.",
        "No weather, gate, crew, or passenger-connection data. One airport, one month of decision testing.",
    ]
    for item in limits:
        story.append(Paragraph(item, bullet, bulletText="•"))

    story.append(Paragraph("What I would do with Delta's internal data", h2))
    internal = [
        "Use the planned tail assignment at decision time, which removes the swap problem and lets turn time back in.",
        "Measure R and H from real spare and swap records instead of assuming them.",
        "Add crew connections, gate availability, and inbound flights from across the network, not just into ATL.",
        "Add weather and air-traffic programs (ground stops, delay programs) known at decision time.",
        "Predict delay minutes directly and recalibrate monthly, since calibration drifts with the delay level.",
    ]
    for item in internal:
        story.append(Paragraph(item, bullet, bulletText="•"))

    doc = SimpleDocTemplate(f"{OUT}/brief.pdf", pagesize=letter,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="DL ATL delay risk and spare aircraft brief")
    doc.build(story)

    pages = len(PdfReader(f"{OUT}/brief.pdf").pages)
    print(f"Wrote {OUT}/brief.pdf ({pages} pages)")
    if pages > 2:
        raise RuntimeError("Brief is longer than 2 pages")


if __name__ == "__main__":
    main()
