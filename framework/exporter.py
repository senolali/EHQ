"""
EHQ-3000 Excel Exporter
==========================
EHQ v1'de kullanilan src/exporter.py'nin portu (5 sekme, ayni renk/
bicim semasi). TEK GENISLEME: v1'de FEQ/PCQ/HNQ olarak sabit kodlanmis
her yer, EHQ-3000'in 4. kategorisi CCQ'yu da kapsayacak sekilde
genellestirildi (CATEGORIES listesi disaridan verilir).
"""

import logging
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger("ehq_exporter")

C_HEADER_DARK  = "1F3864"
C_HEADER_MID   = "2E75B6"
C_ROW_ALT      = "F2F7FC"
C_WHITE        = "FFFFFF"
C_RED_LIGHT    = "FCE4D6"
C_GREEN_LIGHT  = "E2EFDA"
C_BORDER       = "B8CCE4"


def _border(color=C_BORDER):
    s = Side(style="thin", color=color)
    return Border(left=s, right=s, top=s, bottom=s)


def _hdr(bold=True, color=C_WHITE, size=10):
    return Font(name="Arial", bold=bold, color=color, size=size)


def _fill(color):
    return PatternFill("solid", fgColor=color)


def _center():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)


def _left():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)


def _ranked_models(all_results, key="EHQ"):
    models = [(n, d) for n, d in all_results.items()
              if n not in ("__correlation__",) and isinstance(d, dict) and "EHQ" in d]
    models.sort(key=lambda x: x[1].get(key, 0), reverse=True)
    return models


# ─────────────────────────────────────────────────────────────
# SEKME 1: Genel Özet
# ─────────────────────────────────────────────────────────────

def _sheet_summary(wb, all_results, categories):
    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False

    n_cols = 8 + len(categories)
    last_col = get_column_letter(n_cols)
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = "EHQ-3000 Evaluation Results — Summary"
    ws["A1"].font = Font(name="Arial", bold=True, size=13, color=C_WHITE)
    ws["A1"].fill = _fill(C_HEADER_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 28

    headers = ["Model", "Type", "N", "EHQ1", "EHQ2", "EHQ3", "EHQ", "Rank"] + \
              [f"EHQ2_{c}" for c in categories]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = _hdr(); c.fill = _fill(C_HEADER_MID)
        c.alignment = _center(); c.border = _border()
    ws.row_dimensions[2].height = 22

    models = _ranked_models(all_results)
    for rank, (name, d) in enumerate(models, 1):
        row = rank + 2
        cat2 = d.get("EHQ2_by_category", {})
        values = [
            name, d.get("model_type", "-"), d.get("n_questions", 0),
            d.get("EHQ1"), d.get("EHQ2"), d.get("EHQ3"), d.get("EHQ"),
            rank,
        ] + [cat2.get(c) for c in categories]
        fill_color = C_ROW_ALT if rank % 2 == 0 else C_WHITE
        for col, val in enumerate(values, 1):
            c = ws.cell(row=row, column=col)
            c.border = _border(); c.alignment = _center(); c.fill = _fill(fill_color)
            if isinstance(val, float):
                c.value = val; c.number_format = "0.000"
                if col in (4, 5, 6, 7) or col > 8:
                    if val >= 0.75:
                        c.fill = _fill(C_GREEN_LIGHT)
                    elif val < 0.40:
                        c.fill = _fill(C_RED_LIGHT)
            elif val is None:
                c.value = "-"
            else:
                c.value = val
                if col == 1:
                    c.font = Font(name="Arial", bold=True, size=10)
                    c.alignment = _left()
        ws.row_dimensions[row].height = 18

    widths = [22, 10, 6, 8, 8, 8, 8, 6] + [10] * len(categories)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    note_row = len(models) + 4
    ws.merge_cells(f"A{note_row}:{last_col}{note_row}")
    ws[f"A{note_row}"] = (
        "EHQ1=Epistemic Restraint Rate  |  EHQ2=Hallucination Resistance  |  "
        "EHQ3=Confidence-Accuracy Alignment  |  EHQ=Composite (β1=0.30, β2=0.45, β3=0.25)"
    )
    ws[f"A{note_row}"].font = Font(name="Arial", italic=True, size=9, color="595959")
    ws[f"A{note_row}"].alignment = _left()


# ─────────────────────────────────────────────────────────────
# SEKME 2: Kategori Detayı
# ─────────────────────────────────────────────────────────────

def _sheet_categories(wb, all_results, categories):
    ws = wb.create_sheet("Category Detail")
    ws.sheet_view.showGridLines = False

    n_cols = 3 + len(categories)
    last_col = get_column_letter(n_cols)
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = "EHQ2 — Hallucination Resistance by Category"
    ws["A1"].font = Font(name="Arial", bold=True, size=12, color=C_WHITE)
    ws["A1"].fill = _fill(C_HEADER_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 26

    hdrs = ["Model"] + categories + ["Best Category", "Worst Category"]
    for col, h in enumerate(hdrs, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = _hdr(); c.fill = _fill(C_HEADER_MID)
        c.alignment = _center(); c.border = _border()
    ws.row_dimensions[2].height = 20

    models = _ranked_models(all_results)
    for row_i, (name, d) in enumerate(models, 3):
        c2 = d.get("EHQ2_by_category", {})
        vals_map = {c: c2.get(c) for c in categories}
        valid = {k: v for k, v in vals_map.items() if v is not None}
        best = max(valid, key=valid.get) if valid else "-"
        worst = min(valid, key=valid.get) if valid else "-"

        row_vals = [name] + [vals_map[c] for c in categories] + [best, worst]
        fill_c = C_ROW_ALT if row_i % 2 == 0 else C_WHITE
        for col, val in enumerate(row_vals, 1):
            c = ws.cell(row=row_i, column=col)
            c.border = _border(); c.fill = _fill(fill_c); c.alignment = _center()
            if isinstance(val, float):
                c.value = val; c.number_format = "0.000"
                if 2 <= col <= 1 + len(categories):
                    c.fill = _fill(C_GREEN_LIGHT if val >= 0.75 else
                                   (C_RED_LIGHT if val < 0.40 else fill_c))
            elif val is None:
                c.value = "-"
            else:
                c.value = val
                if col == 1:
                    c.font = Font(name="Arial", bold=True, size=10)
                    c.alignment = _left()
        ws.row_dimensions[row_i].height = 18

    widths = [22] + [14] * len(categories) + [14, 14]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ─────────────────────────────────────────────────────────────
# SEKME 3: Response Distribution
# ─────────────────────────────────────────────────────────────

def _sheet_distribution(wb, all_results):
    ws = wb.create_sheet("Response Distribution")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:H1")
    ws["A1"] = "Response Type Distribution (%)"
    ws["A1"].font = Font(name="Arial", bold=True, size=12, color=C_WHITE)
    ws["A1"].fill = _fill(C_HEADER_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 26

    hdrs = ["Model", "N", "ABSTAIN", "HEDGE", "CONFIDENT_CORRECT",
            "CONFIDENT_WRONG", "Appropriate (A+H)", "Hallucination Rate"]
    for col, h in enumerate(hdrs, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = _hdr(); c.fill = _fill(C_HEADER_MID)
        c.alignment = _center(); c.border = _border()
    ws.row_dimensions[2].height = 22

    models = _ranked_models(all_results)
    for row_i, (name, d) in enumerate(models, 3):
        dist = d.get("response_distribution", {})
        ab = dist.get("ABSTAIN", {}).get("pct", 0)
        hg = dist.get("HEDGE", {}).get("pct", 0)
        cc = dist.get("CONFIDENT_CORRECT", {}).get("pct", 0)
        cw = dist.get("CONFIDENT_WRONG", {}).get("pct", 0)
        appropriate = round(ab + hg, 1)
        hallucination = round(cw, 1)

        row_vals = [name, d.get("n_questions", 0), ab, hg, cc, cw,
                    appropriate, hallucination]
        fill_c = C_ROW_ALT if row_i % 2 == 0 else C_WHITE
        for col, val in enumerate(row_vals, 1):
            c = ws.cell(row=row_i, column=col)
            c.border = _border(); c.alignment = _center(); c.fill = _fill(fill_c)
            if col == 1:
                c.value = val
                c.font = Font(name="Arial", bold=True, size=10)
                c.alignment = _left()
            elif isinstance(val, float):
                c.value = val; c.number_format = "0.0\"%\""
                if col == 7:
                    c.fill = _fill(C_GREEN_LIGHT if val >= 50 else
                                   (C_RED_LIGHT if val < 25 else fill_c))
                elif col == 8:
                    c.fill = _fill(C_GREEN_LIGHT if val <= 20 else
                                   (C_RED_LIGHT if val > 40 else fill_c))
            else:
                c.value = val
        ws.row_dimensions[row_i].height = 18

    widths = [22, 6, 10, 10, 18, 18, 16, 18]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ─────────────────────────────────────────────────────────────
# SEKME 4: Kalibrasyon Analizi
# ─────────────────────────────────────────────────────────────

def _sheet_calibration(wb, all_results):
    ws = wb.create_sheet("Calibration Analysis")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:H1")
    ws["A1"] = "EHQ3 — Confidence-Accuracy Alignment (CONFIDENT responses only)"
    ws["A1"].font = Font(name="Arial", bold=True, size=12, color=C_WHITE)
    ws["A1"].fill = _fill(C_HEADER_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 26

    hdrs = ["Model", "N (Confident)", "Avg Confidence", "Avg Accuracy",
            "Gap (Conf−Acc)", "Direction", "EHQ3", "Interpretation"]
    for col, h in enumerate(hdrs, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = _hdr(); c.fill = _fill(C_HEADER_MID)
        c.alignment = _center(); c.border = _border()
    ws.row_dimensions[2].height = 22

    models = _ranked_models(all_results, key="EHQ3")
    for row_i, (name, d) in enumerate(models, 3):
        raw = d.get("raw_results", [])
        conf_res = [r for r in raw
                    if r.get("response_type") in ("CONFIDENT_CORRECT", "CONFIDENT_WRONG")]
        n_conf = len(conf_res)
        avg_conf = (sum(r["confidence"] for r in conf_res) / n_conf) if n_conf else None
        n_corr = sum(1 for r in conf_res if r.get("response_type") == "CONFIDENT_CORRECT")
        avg_acc = n_corr / n_conf if n_conf else None
        gap = round(avg_conf - avg_acc, 3) if avg_conf is not None and avg_acc is not None else None
        direction = ("Overconfident" if gap and gap > 0.1 else
                     "Underconfident" if gap and gap < -0.1 else "Well-calibrated")
        ehq3 = d.get("EHQ3")
        interp = ("Good" if ehq3 and ehq3 >= 0.70 else
                  "Moderate" if ehq3 and ehq3 >= 0.50 else "Poor")

        row_vals = [name, n_conf, avg_conf, avg_acc, gap, direction, ehq3, interp]
        fill_c = C_ROW_ALT if row_i % 2 == 0 else C_WHITE
        for col, val in enumerate(row_vals, 1):
            c = ws.cell(row=row_i, column=col)
            c.border = _border(); c.alignment = _center(); c.fill = _fill(fill_c)
            if col == 1:
                c.value = val
                c.font = Font(name="Arial", bold=True, size=10)
                c.alignment = _left()
            elif isinstance(val, float):
                c.value = val; c.number_format = "0.000"
                if col == 5 and val is not None:
                    c.fill = _fill(C_GREEN_LIGHT if val < 0.10 else
                                   (C_RED_LIGHT if val > 0.30 else fill_c))
                elif col == 7 and val is not None:
                    c.fill = _fill(C_GREEN_LIGHT if val >= 0.70 else
                                   (C_RED_LIGHT if val < 0.50 else fill_c))
            else:
                c.value = val if val is not None else "-"
        ws.row_dimensions[row_i].height = 18

    widths = [22, 14, 16, 14, 16, 16, 8, 14]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ─────────────────────────────────────────────────────────────
# SEKME 5: Model karsilastirma / siralama
# ─────────────────────────────────────────────────────────────
# NOT: v1'deki "EHQ vs CQ" sekmesi, eski 7-modelin CQ (correctness
# quotient) taban degerlerine hardcoded bagliydi (Senol et al. 2026
# baseline). v2'nin 20 modeli buyuk cogunlukla FARKLI isimlere sahip
# (orn. "Claude-4.5-Haiku" != v1'in "Claude-Haiku-4.5"), yani o sabit
# sozluk yeni modellerin cogunu eslestiremez. Bu sekme onun yerine
# NOTR bir "Model Ranking" ozet tablosu -- CQ karsilastirmasi ayri
# bir adimda (elde mevcut CQ verisi olunca) yapilmali.

def _sheet_ranking(wb, all_results):
    ws = wb.create_sheet("Model Ranking")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:F1")
    ws["A1"] = "Model Ranking by EHQ (Composite)"
    ws["A1"].font = Font(name="Arial", bold=True, size=12, color=C_WHITE)
    ws["A1"].fill = _fill(C_HEADER_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 26

    hdrs = ["Rank", "Model", "EHQ1", "EHQ2", "EHQ3", "EHQ"]
    for col, h in enumerate(hdrs, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = _hdr(); c.fill = _fill(C_HEADER_MID)
        c.alignment = _center(); c.border = _border()
    ws.row_dimensions[2].height = 20

    models = _ranked_models(all_results)
    for rank, (name, d) in enumerate(models, 1):
        row_i = rank + 2
        row_vals = [rank, name, d.get("EHQ1"), d.get("EHQ2"), d.get("EHQ3"), d.get("EHQ")]
        fill_c = C_ROW_ALT if row_i % 2 == 0 else C_WHITE
        for col, val in enumerate(row_vals, 1):
            c = ws.cell(row=row_i, column=col)
            c.border = _border(); c.alignment = _center(); c.fill = _fill(fill_c)
            if isinstance(val, float):
                c.value = val; c.number_format = "0.000"
            else:
                c.value = val
                if col == 2:
                    c.font = Font(name="Arial", bold=True, size=10)
                    c.alignment = _left()
        ws.row_dimensions[row_i].height = 18

    widths = [6, 22, 8, 8, 8, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ─────────────────────────────────────────────────────────────
# ANA FONKSIYON
# ─────────────────────────────────────────────────────────────

def export_to_excel(all_results: dict, output_path: str, categories: list) -> str:
    wb = Workbook()
    wb.remove(wb.active)

    _sheet_summary(wb, all_results, categories)
    _sheet_categories(wb, all_results, categories)
    _sheet_distribution(wb, all_results)
    _sheet_calibration(wb, all_results)
    _sheet_ranking(wb, all_results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info("Excel kaydedildi: %s", output_path)
    return output_path
