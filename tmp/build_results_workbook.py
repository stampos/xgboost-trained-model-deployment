import json
import math
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "tmp" / "ml_comparison_results.json"
OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "ml-comparison-20260918"
    / "ml_algorithm_results.xlsx"
)

NAVY = "1F4E78"
BLUE = "D9EAF7"
LIGHT_BLUE = "EAF2F8"
ORANGE = "FCE4D6"
GREEN = "E2F0D9"
GRAY = "F2F2F2"
WHITE = "FFFFFF"
TEXT = "1F1F1F"
GRID = "D9E2F3"

THIN_GRAY = Side(style="thin", color=GRID)
MEDIUM_BLUE = Side(style="medium", color="9EADBA")


def title(ws, cell_range, text):
    ws.merge_cells(cell_range)
    cell = ws[cell_range.split(":")[0]]
    cell.value = text
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[cell.row].height = 32


def section_header(ws, cell_range, text):
    ws.merge_cells(cell_range)
    cell = ws[cell_range.split(":")[0]]
    cell.value = text
    cell.fill = PatternFill("solid", fgColor=BLUE)
    cell.font = Font(name="Aptos", bold=True, color=TEXT)
    cell.alignment = Alignment(horizontal="left", vertical="center")
    for row in ws[cell_range]:
        for item in row:
            item.border = Border(bottom=MEDIUM_BLUE)


def write_rows(ws, start_row, start_col, rows):
    for row_offset, row in enumerate(rows):
        for col_offset, value in enumerate(row):
            ws.cell(start_row + row_offset, start_col + col_offset, value)


def add_table(ws, ref, name, style="TableStyleMedium2"):
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name=style,
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def style_number_range(ws, cell_range, number_format="0.0000"):
    for row in ws[cell_range]:
        for cell in row:
            cell.number_format = number_format


def set_card(ws, cell_range, rows, fill):
    start = ws[cell_range.split(":")[0]]
    write_rows(ws, start.row, start.column, rows)
    for row in ws[cell_range]:
        for cell in row:
            cell.fill = PatternFill("solid", fgColor=fill)
            cell.border = Border(
                left=THIN_GRAY, right=THIN_GRAY, top=THIN_GRAY, bottom=THIN_GRAY
            )
            cell.alignment = Alignment(vertical="center")
    for row in ws[cell_range]:
        for index, cell in enumerate(row):
            if index % 2 == 0:
                cell.font = Font(name="Aptos", bold=True, color=TEXT)


def finalize_sheet(ws, freeze="A4", max_width=28):
    ws.freeze_panes = freeze
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.page_margins.left = 0.25
    ws.page_margins.right = 0.25
    ws.page_margins.top = 0.4
    ws.page_margins.bottom = 0.4
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None and cell.font == Font():
                cell.font = Font(name="Aptos", size=10, color=TEXT)
            cell.alignment = Alignment(
                horizontal=cell.alignment.horizontal,
                vertical="center",
                wrap_text=cell.alignment.wrap_text,
            )
    for column_index, column_cells in enumerate(ws.columns, start=1):
        column_letter = get_column_letter(column_index)
        best = 0
        for cell in column_cells:
            if cell.value is not None:
                best = max(best, len(str(cell.value)))
        ws.column_dimensions[column_letter].width = min(max(best + 2, 10), max_width)
    ws.auto_filter.ref = None


def build_workbook(result):
    wb = Workbook()
    wb.remove(wb.active)
    summary = wb.create_sheet("Summary")
    averages = wb.create_sheet("Average Results")
    folds = wb.create_sheet("Fold Results")
    liquids = wb.create_sheet("XGB by Liquid")
    predictions = wb.create_sheet("XGB Predictions")
    datasets = wb.create_sheet("Datasets")
    config = wb.create_sheet("Configuration")

    # Summary
    title(summary, "A1:M1", "Regression Algorithm Comparison")
    summary.merge_cells("A3:M3")
    summary["A3"] = (
        "5-fold cross-validation | KFold(shuffle=True, random_state=42) | "
        "Lower MAE/RMSE and higher R2 are better"
    )
    summary["A3"].fill = PatternFill("solid", fgColor=GRAY)
    summary["A3"].font = Font(name="Aptos", italic=True, color=TEXT)
    summary["A3"].alignment = Alignment(horizontal="center")

    dataset_one = next(d for d in result["datasets"] if d["dataset"] == "mesos_oros.csv")
    dataset_two = next(
        d for d in result["datasets"] if d["dataset"] == "mesos_oros_nomeli.csv"
    )
    set_card(
        summary,
        "A5:F7",
        [
            ["Dataset", dataset_one["dataset"], "Rows", dataset_one["rows"], "Liquids", dataset_one["liquids"]],
            ["Missing", dataset_one["missing_values"], "Target mean", dataset_one["target_mean"], "Target std", dataset_one["target_std"]],
            ["Target min", dataset_one["target_min"], "Target max", dataset_one["target_max"], "Shapes", dataset_one["shapes"]],
        ],
        LIGHT_BLUE,
    )
    set_card(
        summary,
        "H5:M7",
        [
            ["Dataset", dataset_two["dataset"], "Rows", dataset_two["rows"], "Liquids", dataset_two["liquids"]],
            ["Missing", dataset_two["missing_values"], "Target mean", dataset_two["target_mean"], "Target std", dataset_two["target_std"]],
            ["Target min", dataset_two["target_min"], "Target max", dataset_two["target_max"], "Shapes", dataset_two["shapes"]],
        ],
        ORANGE,
    )
    style_number_range(summary, "B6:F7")
    style_number_range(summary, "I6:M7")

    rank_headers = ["Rank", "Algorithm", "MAE", "RMSE", "R2", "Elapsed (s)"]
    rank_one = sorted(
        (r for r in result["summary_results"] if r["dataset"] == "mesos_oros.csv"),
        key=lambda r: r["mae_mean"],
    )
    rank_two = sorted(
        (
            r
            for r in result["summary_results"]
            if r["dataset"] == "mesos_oros_nomeli.csv"
        ),
        key=lambda r: r["mae_mean"],
    )
    section_header(summary, "A9:F9", "mesos_oros.csv - ranked by MAE")
    section_header(summary, "H9:M9", "mesos_oros_nomeli.csv - ranked by MAE")
    rows_one = [rank_headers] + [
        [
            index,
            row["algorithm"],
            row["mae_mean"],
            row["rmse_mean"],
            row["r2_mean"],
            row["total_elapsed_s"],
        ]
        for index, row in enumerate(rank_one, start=1)
    ]
    rows_two = [rank_headers] + [
        [
            index,
            row["algorithm"],
            row["mae_mean"],
            row["rmse_mean"],
            row["r2_mean"],
            row["total_elapsed_s"],
        ]
        for index, row in enumerate(rank_two, start=1)
    ]
    write_rows(summary, 10, 1, rows_one)
    write_rows(summary, 10, 8, rows_two)
    add_table(summary, "A10:F15", "SummaryRankFull", "TableStyleMedium2")
    add_table(summary, "H10:M15", "SummaryRankNoMeli", "TableStyleMedium9")
    style_number_range(summary, "C11:F15")
    style_number_range(summary, "J11:M15")
    for cell_range in ("A11:F11", "H11:M11"):
        for row in summary[cell_range]:
            for cell in row:
                cell.fill = PatternFill("solid", fgColor=GREEN)
                cell.font = Font(name="Aptos", bold=True, color=TEXT)

    chart_rows = [["Algorithm", "mesos_oros.csv", "mesos_oros_nomeli.csv"]]
    for algorithm in [
        "Linear Regression",
        "Polynomial Ridge",
        "Random Forest",
        "XGBoost",
        "Neural Network",
    ]:
        one = next(
            r
            for r in result["summary_results"]
            if r["dataset"] == "mesos_oros.csv" and r["algorithm"] == algorithm
        )
        two = next(
            r
            for r in result["summary_results"]
            if r["dataset"] == "mesos_oros_nomeli.csv" and r["algorithm"] == algorithm
        )
        chart_rows.append([algorithm, one["r2_mean"], two["r2_mean"]])
    write_rows(summary, 19, 1, chart_rows)
    add_table(summary, "A19:C24", "SummaryR2ChartData", "TableStyleMedium2")
    style_number_range(summary, "B20:C24")
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "Mean R2 by algorithm"
    chart.y_axis.title = "R2"
    chart.x_axis.title = "Algorithm"
    chart.y_axis.scaling.min = -0.25
    chart.y_axis.scaling.max = 1.0
    chart.height = 8.2
    chart.width = 15.5
    chart.add_data(Reference(summary, min_col=2, max_col=3, min_row=19, max_row=24), titles_from_data=True)
    chart.set_categories(Reference(summary, min_col=1, min_row=20, max_row=24))
    summary.add_chart(chart, "E18")

    # Average results
    title(averages, "A1:M1", "Mean Metrics and Standard Deviations by Model")
    average_headers = [
        "Dataset",
        "Rank MAE",
        "Algorithm",
        "MAE mean",
        "MAE std",
        "MSE mean",
        "MSE std",
        "RMSE mean",
        "RMSE std",
        "R2 mean",
        "R2 std",
        "Fit mean (s)",
        "Total elapsed (s)",
    ]
    average_rows = [average_headers]
    for dataset_name in ("mesos_oros.csv", "mesos_oros_nomeli.csv"):
        ranked = sorted(
            (r for r in result["summary_results"] if r["dataset"] == dataset_name),
            key=lambda r: r["mae_mean"],
        )
        for index, row in enumerate(ranked, start=1):
            average_rows.append(
                [
                    dataset_name,
                    index,
                    row["algorithm"],
                    row["mae_mean"],
                    row["mae_std"],
                    row["mse_mean"],
                    row["mse_std"],
                    row["rmse_mean"],
                    row["rmse_std"],
                    row["r2_mean"],
                    row["r2_std"],
                    row["fit_time_mean_s"],
                    row["total_elapsed_s"],
                ]
            )
    write_rows(averages, 3, 1, average_rows)
    add_table(averages, "A3:M13", "AverageResults", "TableStyleMedium2")
    style_number_range(averages, "D4:M13")
    for cell_range in ("D4:D13", "J4:J13"):
        averages.conditional_formatting.add(
            cell_range,
            ColorScaleRule(
                start_type="min",
                start_color="63BE7B",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB84",
                end_type="max",
                end_color="F8696B",
            ),
        )

    # Per-fold results
    title(folds, "A1:I1", "Detailed Results by Fold")
    fold_headers = [
        "Dataset",
        "Algorithm",
        "Fold",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "Fit time (s)",
        "Score time (s)",
    ]
    fold_rows = [fold_headers] + [
        [
            row["dataset"],
            row["algorithm"],
            row["fold"],
            row["mae"],
            row["mse"],
            row["rmse"],
            row["r2"],
            row["fit_time_s"],
            row["score_time_s"],
        ]
        for row in result["fold_results"]
    ]
    write_rows(folds, 3, 1, fold_rows)
    add_table(folds, f"A3:I{3 + len(fold_rows) - 1}", "FoldResults", "TableStyleMedium2")
    style_number_range(folds, f"D4:I{3 + len(fold_rows) - 1}")

    # XGBoost by liquid
    title(liquids, "A1:H1", "XGBoost: Out-of-Fold MAE by Liquid")
    liquid_headers = [
        "Dataset",
        "Liquid",
        "Count",
        "Actual mean (s)",
        "Prediction mean (s)",
        "MAE (s)",
        "Abs. error std (s)",
        "Overall MAE (s)",
    ]
    liquid_rows = [liquid_headers] + [
        [
            row["dataset"],
            row["liquid"],
            row["count"],
            row["actual_mean"],
            row["prediction_mean"],
            row["mae"],
            row["error_std"],
            row["overall_mae"],
        ]
        for row in result["xgboost_liquid_summary"]
    ]
    write_rows(liquids, 3, 1, liquid_rows)
    liquid_end = 3 + len(liquid_rows) - 1
    add_table(liquids, f"A3:H{liquid_end}", "XgbLiquidResults", "TableStyleMedium9")
    style_number_range(liquids, f"D4:H{liquid_end}")
    liquids.conditional_formatting.add(
        f"F4:F{liquid_end}",
        ColorScaleRule(
            start_type="min",
            start_color="63BE7B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="F8696B",
        ),
    )

    # XGBoost predictions
    title(predictions, "A1:K1", "XGBoost: Out-of-Fold Predictions")
    prediction_headers = [
        "Dataset",
        "Row",
        "Fold",
        "Liquid",
        "Height (cm)",
        "Shape",
        "Mass (g)",
        "Actual (s)",
        "Prediction (s)",
        "Error (s)",
        "Absolute error (s)",
    ]
    prediction_rows = [prediction_headers] + [
        [
            row["dataset"],
            row["row"],
            row["fold"],
            row["liquid"],
            row["height_cm"],
            row["shape"],
            row["mass_g"],
            row["actual_s"],
            row["prediction_s"],
            row["error_s"],
            row["absolute_error_s"],
        ]
        for row in result["xgboost_oof_predictions"]
    ]
    write_rows(predictions, 3, 1, prediction_rows)
    prediction_end = 3 + len(prediction_rows) - 1
    add_table(
        predictions,
        f"A3:K{prediction_end}",
        "XgbOofPredictions",
        "TableStyleMedium2",
    )
    style_number_range(predictions, f"E4:E{prediction_end}", "0.00")
    style_number_range(predictions, f"G4:G{prediction_end}", "0.00")
    style_number_range(predictions, f"H4:K{prediction_end}")
    predictions.conditional_formatting.add(
        f"K4:K{prediction_end}",
        ColorScaleRule(
            start_type="min",
            start_color="63BE7B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="F8696B",
        ),
    )

    # Dataset profiles and counts
    title(datasets, "A1:J1", "Dataset Profiles")
    dataset_headers = [
        "Dataset",
        "Rows",
        "Columns",
        "Missing",
        "Liquids",
        "Shapes",
        "Target mean",
        "Target std",
        "Target min",
        "Target max",
    ]
    dataset_rows = [dataset_headers] + [
        [
            row["dataset"],
            row["rows"],
            row["columns"],
            row["missing_values"],
            row["liquids"],
            row["shapes"],
            row["target_mean"],
            row["target_std"],
            row["target_min"],
            row["target_max"],
        ]
        for row in result["datasets"]
    ]
    write_rows(datasets, 3, 1, dataset_rows)
    add_table(datasets, "A3:J5", "DatasetProfiles", "TableStyleMedium2")
    style_number_range(datasets, "G4:J5")
    count_rows = [["Dataset", "Liquid", "Count"]]
    for dataset in result["datasets"]:
        for liquid, count in dataset["liquid_counts"].items():
            count_rows.append([dataset["dataset"], liquid, count])
    write_rows(datasets, 8, 1, count_rows)
    add_table(
        datasets,
        f"A8:C{8 + len(count_rows) - 1}",
        "DatasetLiquidCounts",
        "TableStyleMedium9",
    )

    # Configuration
    title(config, "A1:D1", "Experiment Configuration and Provenance")
    write_rows(
        config,
        3,
        1,
        [
            ["Method", result["cv"]["method"]],
            ["Folds", result["cv"]["n_splits"]],
            ["Shuffle", result["cv"]["shuffle"]],
            ["Random state", result["cv"]["random_state"]],
            ["Target", result["target"]],
        ],
    )
    for row in config["A3:B7"]:
        for cell in row:
            cell.fill = PatternFill("solid", fgColor=GRAY)
    for cell in config["A3:A7"]:
        cell[0].font = Font(name="Aptos", bold=True, color=TEXT)
    config_headers = ["Algorithm / analysis", "Source file", "Parameters", "Note"]
    config_rows = [config_headers]
    for model in result["model_configs"]:
        note = (
            "Additional analysis of the same XGBoost model; not a separate ranked model"
            if model["algorithm"] == "XGBoost MAE per liquid"
            else "5-fold cross-validation"
        )
        config_rows.append(
            [model["algorithm"], model["source_file"], model["parameters"], note]
        )
    write_rows(config, 10, 1, config_rows)
    add_table(config, "A10:D16", "ExperimentConfiguration", "TableStyleMedium2")
    write_rows(
        config,
        18,
        1,
        [
            ["Generated UTC", result["generated_at_utc"]],
            ["Warnings", len(result["warnings"])],
            ["Note", "Metrics come from the actual model executions."],
        ],
    )
    for cell in config["A18:A20"]:
        cell[0].font = Font(name="Aptos", bold=True, color=TEXT)
    for row in config["C11:D16"]:
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row_number in range(11, 17):
        config.row_dimensions[row_number].height = 45

    for sheet in wb.worksheets:
        finalize_sheet(sheet)
    summary.freeze_panes = "A10"
    config.column_dimensions["B"].width = 32
    config.column_dimensions["C"].width = 70
    config.column_dimensions["D"].width = 48
    predictions.column_dimensions["A"].width = 24
    folds.column_dimensions["A"].width = 24
    averages.column_dimensions["A"].width = 24
    liquids.column_dimensions["A"].width = 24
    datasets.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 22
    summary.column_dimensions["I"].width = 22
    summary.sheet_properties.pageSetUpPr.fitToPage = True
    summary.page_setup.fitToWidth = 1
    summary.page_setup.fitToHeight = 1
    return wb, rank_one, rank_two


def verify_workbook(path, expected):
    wb = load_workbook(path, data_only=False)
    expected_sheets = [
        "Summary",
        "Average Results",
        "Fold Results",
        "XGB by Liquid",
        "XGB Predictions",
        "Datasets",
        "Configuration",
    ]
    assert wb.sheetnames == expected_sheets
    assert wb["Average Results"].max_row == 13
    assert wb["Fold Results"].max_row == 53
    assert wb["XGB by Liquid"].max_row == 14
    assert wb["XGB Predictions"].max_row == 393
    assert wb["Summary"]["B11"].value == "XGBoost"
    assert wb["Summary"]["I11"].value == "XGBoost"
    assert len(wb["Summary"]._charts) == 1
    assert len(wb["Average Results"].tables) == 1
    assert len(wb["Fold Results"].tables) == 1
    assert len(wb["XGB by Liquid"].tables) == 1
    assert len(wb["XGB Predictions"].tables) == 1
    assert len(wb["Datasets"].tables) == 2
    assert len(wb["Configuration"].tables) == 1
    formula_errors = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and any(
                    error in cell.value
                    for error in ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A")
                ):
                    formula_errors.append(f"{ws.title}!{cell.coordinate}")
    assert not formula_errors
    assert path.stat().st_size > 50_000
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sheets": wb.sheetnames,
        "summary_rows": len(expected["summary_results"]),
        "fold_rows": len(expected["fold_results"]),
        "liquid_rows": len(expected["xgboost_liquid_summary"]),
        "prediction_rows": len(expected["xgboost_oof_predictions"]),
        "formula_errors": 0,
        "best_full": wb["Summary"]["B11"].value,
        "best_nomeli": wb["Summary"]["I11"].value,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    workbook, _, _ = build_workbook(result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT_PATH)
    verification = verify_workbook(OUTPUT_PATH, result)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
