"""
Generates data/hr_policies_images.pdf — a PDF containing HR policy
content rendered entirely as images (drawings/charts/infographics).
No plain text paragraphs — everything is embedded as a graphic.

Run: python create_hr_images_pdf.py
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line, Circle, Polygon, Group
)
from reportlab.graphics import renderPDF
from reportlab.platypus import Flowable

OUTPUT = "data/hr_policies_images.pdf"
W_PAGE, H_PAGE = A4
MARGIN = 1.8 * cm
W = W_PAGE - 2 * MARGIN

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1A3C6B")
BLUE   = colors.HexColor("#2563EB")
SKY    = colors.HexColor("#DBEAFE")
TEAL   = colors.HexColor("#0D9488")
LTEAL  = colors.HexColor("#CCFBF1")
AMBER  = colors.HexColor("#D97706")
LAMBER = colors.HexColor("#FEF3C7")
RED    = colors.HexColor("#DC2626")
LRED   = colors.HexColor("#FEE2E2")
GREEN  = colors.HexColor("#16A34A")
LGREEN = colors.HexColor("#DCFCE7")
GRAY   = colors.HexColor("#64748B")
LGRAY  = colors.HexColor("#F1F5F9")
WHITE  = colors.white
BLACK  = colors.HexColor("#1E293B")


class DrawingFlowable(Flowable):
    """Wraps a ReportLab Drawing so it flows in a Platypus story."""
    def __init__(self, drawing):
        Flowable.__init__(self)
        self.drawing = drawing
        self.width = drawing.width
        self.height = drawing.height

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def txt(d, x, y, s, size=10, color=BLACK, bold=False, align="left"):
    font = "Helvetica-Bold" if bold else "Helvetica"
    anchor = {"left": "start", "center": "middle", "right": "end"}.get(align, "start")
    d.add(String(x, y, s, fontSize=size, fillColor=color,
                 fontName=font, textAnchor=anchor))


def hline(d, x1, y1, x2, color=GRAY, width=0.8):
    d.add(Line(x1, y1, x2, y1, strokeColor=color, strokeWidth=width))


# ── 1. Cover Banner ──────────────────────────────────────────────────────────

def cover_banner(w=W, h=3.5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=NAVY, strokeColor=None))
    # Decorative circles
    for cx, cy, r, alpha in [(w*0.88, h*0.5, 55, 0.15), (w*0.93, h*0.2, 30, 0.10)]:
        d.add(Circle(cx, cy, r, fillColor=WHITE, strokeColor=None,
                     fillOpacity=alpha, strokeOpacity=0))
    txt(d, w/2, h*0.62, "ACME CORPORATION", 22, WHITE, bold=True, align="center")
    txt(d, w/2, h*0.28, "HR POLICY QUICK-REFERENCE  ·  Effective January 2025",
        10, SKY, align="center")
    return d


# ── 2. Leave Entitlement Bar Chart ───────────────────────────────────────────

def leave_bar_chart(w=W, h=7*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=LGRAY, strokeColor=None))

    # Title
    txt(d, w/2, h - 0.55*cm, "Annual Leave Days by Grade & Tenure",
        12, NAVY, bold=True, align="center")

    labels  = ["0–1 yr", "1–3 yrs", "3–5 yrs", "5–10 yrs", "10+ yrs"]
    grades  = {"Senior (A)": (BLUE,  [15, 18, 22, 25, 30]),
               "Mid (B)":    (TEAL,  [12, 15, 18, 22, 25]),
               "Junior (C)": (AMBER, [10, 12, 15, 18, 20])}

    n_groups = len(labels)
    n_bars   = len(grades)
    chart_x  = 1.8*cm
    chart_y  = 1.2*cm
    chart_w  = w - 2.4*cm
    chart_h  = h - 2.0*cm
    max_val  = 32
    group_w  = chart_w / n_groups
    bar_w    = group_w * 0.22
    gap      = group_w * 0.04

    # Y grid + labels
    for v in range(0, 35, 5):
        yy = chart_y + (v / max_val) * chart_h
        hline(d, chart_x, yy, chart_x + chart_w,
              color=colors.HexColor("#CBD5E1"), width=0.5)
        txt(d, chart_x - 0.15*cm, yy - 4, str(v), 7, GRAY, align="right")

    # Bars
    for gi, (grade, (clr, vals)) in enumerate(grades.items()):
        for bi, val in enumerate(vals):
            bx = chart_x + bi * group_w + gi * (bar_w + gap) + gap
            bh = (val / max_val) * chart_h
            d.add(Rect(bx, chart_y, bar_w, bh, fillColor=clr, strokeColor=None))
            txt(d, bx + bar_w/2, chart_y + bh + 2, str(val), 6, clr, align="center")

    # X-axis labels
    for bi, lbl in enumerate(labels):
        cx = chart_x + bi * group_w + group_w / 2
        txt(d, cx, chart_y - 0.4*cm, lbl, 7, GRAY, align="center")

    # Legend
    lx = chart_x
    for grade, (clr, _) in grades.items():
        d.add(Rect(lx, 0.15*cm, 0.35*cm, 0.25*cm, fillColor=clr, strokeColor=None))
        txt(d, lx + 0.45*cm, 0.18*cm, grade, 7, BLACK)
        lx += 2.6*cm

    return d


# ── 3. Leave Types Icon Grid ─────────────────────────────────────────────────

def leave_types_grid(w=W, h=5.5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=WHITE, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Other Leave Types at a Glance",
        12, NAVY, bold=True, align="center")

    items = [
        ("Sick Leave",        "12 days / yr",  LRED,   RED),
        ("Maternity",         "16 weeks",      LTEAL,  TEAL),
        ("Paternity",         "4 weeks",       SKY,    BLUE),
        ("Bereavement",       "5 days",        LAMBER, AMBER),
        ("Study / Exam",      "3 days/exam",   LGREEN, GREEN),
        ("Unpaid Leave",      "Up to 30 days", LGRAY,  GRAY),
    ]

    cols = 3
    card_w = (w - 0.6*cm) / cols
    card_h = (h - 1.2*cm) / 2 - 0.15*cm
    for i, (name, value, bg, fg) in enumerate(items):
        row = i // cols
        col = i %  cols
        cx = col * card_w + 0.3*cm
        cy = h - 1.3*cm - row * (card_h + 0.2*cm) - card_h
        d.add(Rect(cx, cy, card_w - 0.2*cm, card_h,
                   fillColor=bg, strokeColor=fg, strokeWidth=1, rx=6, ry=6))
        mid_y = cy + card_h / 2
        txt(d, cx + (card_w-0.2*cm)/2, mid_y + 0.15*cm, name,
            9, fg, bold=True, align="center")
        txt(d, cx + (card_w-0.2*cm)/2, mid_y - 0.35*cm, value,
            11, BLACK, bold=True, align="center")
    return d


# ── 4. Salary Band Horizontal Bar ───────────────────────────────────────────

def salary_bands(w=W, h=6.5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=LGRAY, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Salary Bands & Bonus Targets",
        12, NAVY, bold=True, align="center")

    bands = [
        ("L1 – Associate",          35,  50,  "5%"),
        ("L2 – Analyst/Engineer",   50,  75,  "8%"),
        ("L3 – Senior Analyst",     75,  100, "10%"),
        ("L4 – Lead/Manager",       100, 140, "15%"),
        ("L5 – Director",           140, 200, "20%"),
        ("L6 – VP/C-Suite",         200, 280, "30%+"),
    ]

    label_w = 4.2*cm
    bonus_w = 1.0*cm
    bar_area_w = w - label_w - bonus_w - 0.6*cm
    max_val = 280
    row_h = (h - 1.3*cm) / len(bands)
    bar_h = row_h * 0.55
    clrs = [BLUE, TEAL, GREEN, AMBER, RED, NAVY]

    for i, (label, lo, hi, bonus) in enumerate(bands):
        y = h - 1.3*cm - (i + 1) * row_h + (row_h - bar_h) / 2
        bx_start = label_w + 0.3*cm + (lo / max_val) * bar_area_w
        bx_len   = ((hi - lo) / max_val) * bar_area_w
        d.add(Rect(label_w + 0.3*cm, y, bar_area_w, bar_h,
                   fillColor=colors.HexColor("#E2E8F0"), strokeColor=None))
        d.add(Rect(bx_start, y, bx_len, bar_h, fillColor=clrs[i], strokeColor=None))
        txt(d, label_w + 0.15*cm, y + bar_h*0.28, label, 8, BLACK, align="right")
        txt(d, bx_start + bx_len/2, y + bar_h*0.22,
            f"${lo}k – ${hi}k", 7, WHITE, bold=True, align="center")
        txt(d, w - 0.3*cm, y + bar_h*0.22, bonus, 8, clrs[i], bold=True, align="right")

    # Axis
    hline(d, label_w + 0.3*cm, 0.5*cm, w - bonus_w - 0.3*cm, GRAY, 0.6)
    for v in [0, 50, 100, 150, 200, 250]:
        ax = label_w + 0.3*cm + (v / max_val) * bar_area_w
        txt(d, ax, 0.15*cm, f"${v}k", 6, GRAY, align="center")

    txt(d, w - 0.15*cm, 0.6*cm, "Bonus", 7, GRAY, align="right")
    return d


# ── 5. Performance Rating Gauge Row ─────────────────────────────────────────

def performance_gauges(w=W, h=4.5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=WHITE, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Performance Rating Scale",
        12, NAVY, bold=True, align="center")

    ratings = [
        ("5", "Exceptional",        GREEN,  "1.5×", "Up to 15%"),
        ("4", "Exceeds Exp.",        TEAL,   "1.2×", "Up to 10%"),
        ("3", "Meets Exp.",          BLUE,   "1.0×", "Up to 5%"),
        ("2", "Below Exp.",          AMBER,  "0.5×", "0%"),
        ("1", "Unsatisfactory",      RED,    "0×",   "0% + PIP"),
    ]

    card_w = (w - 0.4*cm) / len(ratings)
    card_h = h - 1.2*cm
    for i, (score, label, clr, mult, raise_) in enumerate(ratings):
        cx = i * card_w + 0.2*cm
        cy = 0.5*cm
        d.add(Rect(cx, cy, card_w - 0.15*cm, card_h,
                   fillColor=clr, strokeColor=None, rx=6, ry=6))
        mid = cx + (card_w - 0.15*cm) / 2
        txt(d, mid, cy + card_h - 0.45*cm, score, 22, WHITE, bold=True, align="center")
        txt(d, mid, cy + card_h - 0.9*cm, label, 7, WHITE, align="center")
        hline(d, cx + 0.2*cm, cy + card_h - 1.05*cm,
              cx + card_w - 0.35*cm, WHITE, 0.5)
        txt(d, mid, cy + 0.65*cm, f"Bonus {mult}", 7, WHITE, bold=True, align="center")
        txt(d, mid, cy + 0.3*cm,  f"Raise: {raise_}", 6.5, WHITE, align="center")
    return d


# ── 6. Org Chart ─────────────────────────────────────────────────────────────

def org_chart(w=W, h=6*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=SKY, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Top-Level Organisational Chart",
        12, NAVY, bold=True, align="center")

    bw, bh = 3.2*cm, 1.0*cm

    def box(cx, cy, label, sublabel="", clr=NAVY):
        d.add(Rect(cx - bw/2, cy - bh/2, bw, bh,
                   fillColor=clr, strokeColor=WHITE, strokeWidth=1.2, rx=5, ry=5))
        txt(d, cx, cy + 0.05*cm, label, 9, WHITE, bold=True, align="center")
        if sublabel:
            txt(d, cx, cy - 0.28*cm, sublabel, 7, SKY, align="center")

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=NAVY, strokeWidth=1.2))
        # Arrowhead
        d.add(Polygon([x2-3, y2+5, x2+3, y2+5, x2, y2],
                      fillColor=NAVY, strokeColor=None))

    # CEO
    ceo_x, ceo_y = w/2, h - 1.8*cm
    box(ceo_x, ceo_y, "CEO", "Sarah Mitchell")

    # L2 nodes
    l2 = [(w*0.18, h - 3.8*cm, "CFO",  "Tom Reid"),
          (w*0.40, h - 3.8*cm, "CHRO", "Lisa Park"),
          (w*0.62, h - 3.8*cm, "CTO",  "Dev Anand"),
          (w*0.84, h - 3.8*cm, "CMO",  "Jane Wu")]
    for (lx, ly, t, s) in l2:
        arrow(ceo_x, ceo_y - bh/2, lx, ly + bh/2)
        box(lx, ly, t, s, BLUE)

    # One L3 under CHRO
    hr_x, hr_y = w*0.40, h - 5.5*cm
    arrow(w*0.40, h - 3.8*cm - bh/2, hr_x, hr_y + bh/2)
    box(hr_x, hr_y, "HR Manager", "Rita Nair", TEAL)
    return d


# ── 7. Remote Work Policy Infographic ───────────────────────────────────────

def remote_work_infographic(w=W, h=5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=LGRAY, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Remote Work Policy — Key Rules",
        12, NAVY, bold=True, align="center")

    rules = [
        (BLUE,  "3 days/week", "Max remote days\n(manager approval)"),
        (TEAL,  "30+ days",    "International remote:\nnotify HR & Legal"),
        (NAVY,  "VP sign-off", "Fully remote\narrangements"),
        (GREEN, "$500",        "One-time remote\nsetup stipend"),
    ]

    card_w = (w - 0.4*cm) / len(rules)
    card_h = h - 1.4*cm
    for i, (clr, big, small) in enumerate(rules):
        cx = i * card_w + 0.2*cm
        cy = 0.7*cm
        d.add(Rect(cx, cy, card_w - 0.15*cm, card_h,
                   fillColor=WHITE, strokeColor=clr, strokeWidth=2, rx=8, ry=8))
        mid = cx + (card_w - 0.15*cm) / 2
        # Coloured top stripe
        d.add(Rect(cx, cy + card_h - 0.6*cm, card_w - 0.15*cm, 0.6*cm,
                   fillColor=clr, strokeColor=None, rx=6, ry=6))
        txt(d, mid, cy + card_h - 0.38*cm, big, 9, WHITE, bold=True, align="center")
        lines = small.split("\n")
        for j, ln in enumerate(lines):
            txt(d, mid, cy + card_h - 1.0*cm - j * 0.35*cm, ln, 7.5, BLACK, align="center")
    return d


# ── 8. Benefits Summary Donut-style tiles ───────────────────────────────────

def benefits_tiles(w=W, h=5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=WHITE, strokeColor=None))
    txt(d, w/2, h - 0.5*cm, "Benefits at a Glance",
        12, NAVY, bold=True, align="center")

    items = [
        ("Health",        "Employee\n+ Family",   "10% premium",   BLUE,  SKY),
        ("Dental/Vision", "Employee\nonly",        "20% premium",   TEAL,  LTEAL),
        ("Life Ins.",     "3× annual\nsalary",     "Fully paid",    GREEN, LGREEN),
        ("401(k)",        "6% employer\nmatch",    "Min 3% to qualify", AMBER, LAMBER),
        ("L&D Budget",    "$2,000\nper year",      "Approved courses", NAVY, LGRAY),
    ]

    cols = 5
    card_w = (w - 0.3*cm) / cols
    card_h = h - 1.2*cm
    for i, (name, val, note, fg, bg) in enumerate(items):
        cx = i * card_w + 0.15*cm
        cy = 0.5*cm
        d.add(Rect(cx, cy, card_w - 0.1*cm, card_h,
                   fillColor=bg, strokeColor=fg, strokeWidth=1.2, rx=8, ry=8))
        mid = cx + (card_w - 0.1*cm) / 2
        txt(d, mid, cy + card_h - 0.45*cm, name, 8, fg, bold=True, align="center")
        hline(d, cx + 0.3*cm, cy + card_h - 0.62*cm,
              cx + card_w - 0.4*cm, fg, 0.5)
        lines = val.split("\n")
        for j, ln in enumerate(lines):
            txt(d, mid, cy + card_h - 0.95*cm - j*0.32*cm, ln, 8, BLACK, bold=True, align="center")
        note_lines = note.split(" ")
        note_str = note if len(note) <= 18 else note[:18] + "…"
        txt(d, mid, cy + 0.25*cm, note_str, 6.5, GRAY, align="center")
    return d


# ── 9. Disciplinary Process Flow ────────────────────────────────────────────

def disciplinary_flow(w=W, h=4.5*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=LGRAY, strokeColor=None))
    txt(d, w/2, h - 0.48*cm, "Disciplinary Process — 3-Level Framework",
        12, NAVY, bold=True, align="center")

    steps = [
        (RED,   "SERIOUS",    "Harassment\nFraud · Violence",   "Immediate\nTermination"),
        (AMBER, "MODERATE",   "Insubordination\nMisuse of assets", "Final Warning\n→ Suspension"),
        (TEAL,  "MINOR",      "Tardiness\nDress code",           "Verbal\n→ Written Warning"),
    ]

    bw = (w - 0.6*cm) / len(steps)
    bh = h - 1.3*cm
    arrow_w = 0.5*cm

    for i, (clr, level, examples, consequence) in enumerate(steps):
        bx = i * bw + 0.3*cm
        by = 0.7*cm
        d.add(Rect(bx, by, bw - 0.2*cm, bh,
                   fillColor=clr, strokeColor=None, rx=6, ry=6))
        mid = bx + (bw - 0.2*cm) / 2
        txt(d, mid, by + bh - 0.38*cm, level, 9, WHITE, bold=True, align="center")
        hline(d, bx + 0.3*cm, by + bh - 0.55*cm, bx + bw - 0.5*cm, WHITE, 0.5)
        ex_lines = examples.split("\n")
        for j, ln in enumerate(ex_lines):
            txt(d, mid, by + bh - 0.95*cm - j * 0.32*cm, ln, 7, WHITE, align="center")
        hline(d, bx + 0.3*cm, by + 0.9*cm, bx + bw - 0.5*cm, WHITE, 0.5)
        cons_lines = consequence.split("\n")
        for j, ln in enumerate(cons_lines):
            txt(d, mid, by + 0.65*cm - j * 0.3*cm, ln, 7.5, WHITE, bold=True, align="center")
    return d


# ── 10. Footer Banner ────────────────────────────────────────────────────────

def footer_banner(w=W, h=1.2*cm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=NAVY, strokeColor=None))
    txt(d, w/2, h*0.38, "Acme Corporation  ·  hr@acmecorp.com  ·  Version 4.0  ·  January 2025",
        8, SKY, align="center")
    return d


# ── Assemble ─────────────────────────────────────────────────────────────────

def build():
    import os
    os.makedirs("data", exist_ok=True)

    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    sp = lambda n=0.3: Spacer(1, n * cm)

    story = [
        DrawingFlowable(cover_banner()),           sp(0.5),
        DrawingFlowable(org_chart()),              sp(0.4),
        DrawingFlowable(leave_bar_chart()),        sp(0.4),
        DrawingFlowable(leave_types_grid()),       sp(0.4),
        DrawingFlowable(salary_bands()),           sp(0.4),
        DrawingFlowable(performance_gauges()),     sp(0.4),
        DrawingFlowable(benefits_tiles()),         sp(0.4),
        DrawingFlowable(remote_work_infographic()),sp(0.4),
        DrawingFlowable(disciplinary_flow()),      sp(0.4),
        DrawingFlowable(footer_banner()),
    ]

    doc.build(story)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    build()
