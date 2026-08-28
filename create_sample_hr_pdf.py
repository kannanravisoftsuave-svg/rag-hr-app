"""
Generates data/hr_document.pdf — a realistic HR policy document
containing headings, paragraphs, tables, and an embedded image.
Run: python create_sample_hr_pdf.py
"""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics import renderPDF

OUTPUT = "data/hr_document.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
BRAND   = colors.HexColor("#1A3C6B")   # dark navy
ACCENT  = colors.HexColor("#E8F0FE")   # light blue fill
HEADER  = colors.HexColor("#2563EB")   # medium blue
RED     = colors.HexColor("#DC2626")
GREEN   = colors.HexColor("#16A34A")
LGRAY   = colors.HexColor("#F1F5F9")
DGRAY   = colors.HexColor("#334155")

def logo_image(width=5*cm, height=1.8*cm):
    """Draw a simple company logo as a ReportLab Drawing."""
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=BRAND, strokeColor=None))
    d.add(Circle(height/2, height/2, height/2 - 4, fillColor=ACCENT, strokeColor=None))
    d.add(String(height + 6, height/2 - 5, "ACME Corp", fontSize=14,
                 fillColor=colors.white, fontName="Helvetica-Bold"))
    d.add(String(height + 6, 4, "Human Resources", fontSize=7,
                 fillColor=ACCENT, fontName="Helvetica"))
    return d

def org_chart_image(width=14*cm, height=5*cm):
    """Draw a simple org-chart diagram."""
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=LGRAY, strokeColor=None))

    boxes = [
        (width/2 - 2.5*cm, height - 1.8*cm, "CEO"),
        (1.5*cm,            height - 4.0*cm, "CFO"),
        (width/2 - 2*cm,    height - 4.0*cm, "CHRO"),
        (width - 5.5*cm,    height - 4.0*cm, "CTO"),
    ]
    bw, bh = 4.5*cm, 1.1*cm
    for x, y, label in boxes:
        d.add(Rect(x, y, bw, bh, fillColor=BRAND, strokeColor=None, rx=4, ry=4))
        d.add(String(x + bw/2 - len(label)*3.5, y + bh/2 - 5,
                     label, fontSize=10, fillColor=colors.white,
                     fontName="Helvetica-Bold"))

    # Lines from CEO down
    cx = width/2 - 2.5*cm + bw/2
    cy_ceo = height - 1.8*cm
    for x, y, _ in boxes[1:]:
        d.add(reportlab_line(cx, cy_ceo, x + bw/2, y + bh,
                             strokeColor=HEADER, strokeWidth=1.2))
    return d

def reportlab_line(x1, y1, x2, y2, **kw):
    from reportlab.graphics.shapes import Line
    return Line(x1, y1, x2, y2, **kw)

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                    textColor=BRAND, fontSize=18, spaceAfter=6,
                    fontName="Helvetica-Bold")
h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                    textColor=HEADER, fontSize=13, spaceBefore=14, spaceAfter=4,
                    fontName="Helvetica-Bold")
h3 = ParagraphStyle("H3", parent=styles["Heading3"],
                    textColor=DGRAY, fontSize=11, spaceBefore=8, spaceAfter=3,
                    fontName="Helvetica-Bold")
body = ParagraphStyle("Body", parent=styles["Normal"],
                      fontSize=10, leading=15, textColor=DGRAY,
                      spaceAfter=6)
note = ParagraphStyle("Note", parent=body,
                      backColor=ACCENT, borderPad=6, fontSize=9,
                      leftIndent=8, rightIndent=8)

def tbl_style(has_header=True):
    base = [
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
    ]
    if has_header:
        base += [
            ("BACKGROUND", (0,0), (-1,0), BRAND),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTSIZE",   (0,0), (-1,0), 9),
        ]
    return TableStyle(base)

# ── Document content ──────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []
    W = A4[0] - 5*cm   # usable width

    # ── Cover / Header ────────────────────────────────────────────────────────
    story.append(RLImage(logo_image(), width=5*cm, height=1.8*cm))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Employee Handbook — HR Policy Document", h1))
    story.append(Paragraph("Acme Corporation · Effective Date: January 1, 2025 · Version 4.0", body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 0.5*cm))

    # ── Section 1: Introduction ───────────────────────────────────────────────
    story.append(Paragraph("1. Introduction", h2))
    story.append(Paragraph(
        "This Employee Handbook sets out the policies, procedures, and guidelines that govern "
        "employment at Acme Corporation. All employees are expected to read, understand, and "
        "comply with the provisions contained herein. This document supersedes all previous "
        "versions of the employee handbook.", body))
    story.append(Paragraph(
        "<b>Note:</b> This handbook does not constitute a contract of employment. Acme Corporation "
        "reserves the right to amend these policies at any time with reasonable notice.",
        note))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 2: Org Structure with image ───────────────────────────────────
    story.append(Paragraph("2. Organisational Structure", h2))
    story.append(Paragraph(
        "The company is led by the Chief Executive Officer (CEO) and organised into three primary "
        "divisions: Finance (CFO), Human Resources (CHRO), and Technology (CTO). "
        "The chart below illustrates the top-level reporting structure.", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(RLImage(org_chart_image(), width=W, height=5*cm))
    story.append(Paragraph("Figure 1 – Top-Level Organisational Chart (as of 2025)",
                            ParagraphStyle("cap", parent=body, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*cm))

    # ── Section 3: Leave Policy with table ───────────────────────────────────
    story.append(Paragraph("3. Leave & Time-Off Policy", h2))
    story.append(Paragraph("3.1 Annual Leave Entitlement", h3))
    story.append(Paragraph(
        "Leave entitlement is based on the employee's length of service and employment grade. "
        "The table below shows the annual leave days per year:", body))

    leave_data = [
        ["Years of Service", "Grade A (Senior)", "Grade B (Mid)", "Grade C (Junior)"],
        ["0 – 1 year",        "15 days",          "12 days",       "10 days"],
        ["1 – 3 years",       "18 days",          "15 days",       "12 days"],
        ["3 – 5 years",       "22 days",          "18 days",       "15 days"],
        ["5 – 10 years",      "25 days",          "22 days",       "18 days"],
        ["10+ years",         "30 days",          "25 days",       "20 days"],
    ]
    t = Table(leave_data, colWidths=[W*0.3, W*0.23, W*0.23, W*0.24])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("3.2 Other Leave Types", h3))
    other_leave = [
        ["Leave Type",           "Entitlement",    "Paid?",  "Notes"],
        ["Sick Leave",           "12 days/year",   "Yes",    "Medical certificate required after 3 days"],
        ["Maternity Leave",      "16 weeks",       "Yes",    "Applicable to primary caregiver"],
        ["Paternity Leave",      "4 weeks",        "Yes",    "Within 6 months of birth"],
        ["Bereavement Leave",    "5 days",         "Yes",    "Immediate family members"],
        ["Study / Exam Leave",   "3 days/exam",    "Yes",    "Approved courses only"],
        ["Unpaid Leave",         "Up to 30 days",  "No",     "Manager + HR approval required"],
        ["Public Holidays",      "As per calendar","Yes",    "11 national + 2 state holidays"],
    ]
    t2 = Table(other_leave, colWidths=[W*0.25, W*0.20, W*0.12, W*0.43])
    t2.setStyle(tbl_style())
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # ── Section 4: Compensation ───────────────────────────────────────────────
    story.append(Paragraph("4. Compensation & Benefits", h2))
    story.append(Paragraph("4.1 Salary Bands", h3))
    story.append(Paragraph(
        "Salaries are structured in bands aligned to market benchmarks. "
        "Annual reviews take place in January. Merit increases are performance-linked.", body))

    salary_data = [
        ["Band", "Role Level",         "Annual Salary Range (USD)", "Bonus Target"],
        ["L1",   "Associate",          "$35,000 – $50,000",         "5%"],
        ["L2",   "Analyst / Engineer", "$50,000 – $75,000",         "8%"],
        ["L3",   "Senior Analyst",     "$75,000 – $100,000",        "10%"],
        ["L4",   "Lead / Manager",     "$100,000 – $140,000",       "15%"],
        ["L5",   "Director",           "$140,000 – $200,000",       "20%"],
        ["L6",   "VP / C-Suite",       "$200,000+",                 "30%+"],
    ]
    t3 = Table(salary_data, colWidths=[W*0.08, W*0.27, W*0.37, W*0.28])
    t3.setStyle(tbl_style())
    story.append(t3)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("4.2 Benefits Summary", h3))
    benefits_data = [
        ["Benefit",                 "Coverage",                    "Employee Contribution"],
        ["Health Insurance",        "Employee + Family",           "10% of premium"],
        ["Dental & Vision",         "Employee only",               "20% of premium"],
        ["Life Insurance",          "3× annual salary",            "Nil (fully employer-paid)"],
        ["401(k) / Pension",        "Up to 6% employer match",     "Min. 3% to qualify"],
        ["Gym / Wellness",          "$600 annual allowance",       "Nil"],
        ["Learning & Development",  "$2,000 annual budget",        "Nil"],
        ["Remote Work Stipend",     "$500 one-time setup",         "Nil"],
    ]
    t4 = Table(benefits_data, colWidths=[W*0.30, W*0.38, W*0.32])
    t4.setStyle(tbl_style())
    story.append(t4)
    story.append(Spacer(1, 0.4*cm))

    # ── Section 5: Performance Review ────────────────────────────────────────
    story.append(Paragraph("5. Performance Management", h2))
    story.append(Paragraph(
        "Acme follows a bi-annual performance review cycle. All employees receive a mid-year "
        "check-in in July and a formal annual review in December. Ratings are on a 5-point scale:", body))

    rating_data = [
        ["Rating", "Label",                 "Bonus Multiplier", "Salary Increase Cap"],
        ["5",      "Exceptional",           "1.5×",             "Up to 15%"],
        ["4",      "Exceeds Expectations",  "1.2×",             "Up to 10%"],
        ["3",      "Meets Expectations",    "1.0×",             "Up to 5%"],
        ["2",      "Below Expectations",    "0.5×",             "0%"],
        ["1",      "Unsatisfactory",        "0×",               "0% (PIP initiated)"],
    ]
    t5 = Table(rating_data, colWidths=[W*0.10, W*0.32, W*0.28, W*0.30])
    t5.setStyle(tbl_style())
    story.append(t5)
    story.append(Spacer(1, 0.4*cm))

    # ── Section 6: Code of Conduct ────────────────────────────────────────────
    story.append(Paragraph("6. Code of Conduct & Disciplinary Policy", h2))
    story.append(Paragraph(
        "All employees are expected to maintain the highest standards of professional conduct. "
        "Violations are categorised into three levels:", body))

    conduct_data = [
        ["Level",    "Examples",                                          "Consequence"],
        ["Minor",    "Tardiness, dress code violations",                  "Verbal warning → Written warning"],
        ["Moderate", "Insubordination, misuse of company assets",         "Final written warning → Suspension"],
        ["Serious",  "Harassment, fraud, data breach, violence",          "Immediate termination"],
    ]
    t6 = Table(conduct_data, colWidths=[W*0.12, W*0.52, W*0.36])
    t6.setStyle(tbl_style())
    story.append(t6)
    story.append(Spacer(1, 0.4*cm))

    # ── Section 7: Remote Work Policy ────────────────────────────────────────
    story.append(Paragraph("7. Remote Work Policy", h2))
    story.append(Paragraph(
        "Effective January 2025, Acme operates on a hybrid-first model. Employees may work "
        "remotely up to 3 days per week subject to manager approval and role eligibility. "
        "Fully remote arrangements require VP-level sign-off and a formal remote work agreement.", body))
    story.append(Paragraph(
        "<b>Important:</b> Employees working remotely internationally for more than 30 consecutive days "
        "must notify HR and Legal to assess tax and compliance obligations.",
        note))
    story.append(Spacer(1, 0.4*cm))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Acme Corporation HR Department · hr@acmecorp.com · Last updated: January 2025 · "
        "For the most current version always refer to the HR intranet portal.",
        ParagraphStyle("footer", parent=body, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    print(f"Created: {OUTPUT}")

if __name__ == "__main__":
    build()
