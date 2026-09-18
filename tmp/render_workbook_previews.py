import sys
from pathlib import Path

import matplotlib.pyplot as plt
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATH = ROOT / "outputs" / "ml-comparison-20260918" / "ml_algorithm_results.xlsx"
PREVIEW_DIR = ROOT / "tmp" / "workbook_previews"


def display_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table_preview(ax, ws, title, min_row, max_row, min_col, max_col, font_size=7):
    rows = []
    for row in ws.iter_rows(
        min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col
    ):
        rows.append([display_value(cell.value) for cell in row])
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#1F4E78")
    table = ax.table(cellText=rows, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2F3")
        if row == 0:
            cell.set_facecolor("#1F4E78")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F9FC")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(WORKBOOK_PATH, data_only=True)

    summary = wb["Summary"]
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.55, 1.1, 1.6])
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.7,
        summary["A1"].value,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#1F4E78", edgecolor="#1F4E78"),
    )
    title_ax.text(0.5, 0.15, summary["A3"].value, ha="center", va="center", fontsize=10)

    left_ax = fig.add_subplot(grid[1, 0])
    right_ax = fig.add_subplot(grid[1, 1])
    table_preview(left_ax, summary, "mesos_oros.csv ranking", 10, 15, 1, 6, 8)
    table_preview(right_ax, summary, "mesos_oros_nomeli.csv ranking", 10, 15, 8, 13, 8)

    helper_ax = fig.add_subplot(grid[2, 0])
    table_preview(helper_ax, summary, "R2 comparison data", 19, 24, 1, 3, 8)
    chart_ax = fig.add_subplot(grid[2, 1])
    algorithms = [summary.cell(row, 1).value for row in range(20, 25)]
    full = [summary.cell(row, 2).value for row in range(20, 25)]
    nomeli = [summary.cell(row, 3).value for row in range(20, 25)]
    positions = list(range(len(algorithms)))
    chart_ax.bar([p - 0.18 for p in positions], full, width=0.36, label="mesos_oros.csv", color="#5B9BD5")
    chart_ax.bar([p + 0.18 for p in positions], nomeli, width=0.36, label="mesos_oros_nomeli.csv", color="#ED7D31")
    chart_ax.set_xticks(positions, algorithms, rotation=25, ha="right")
    chart_ax.set_ylim(-0.25, 1.0)
    chart_ax.set_ylabel("Mean R2")
    chart_ax.set_title("Mean R2 by algorithm", color="#1F4E78", fontweight="bold")
    chart_ax.grid(axis="y", alpha=0.25)
    chart_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "summary_preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    previews = [
        ("Average Results", 3, 10, 1, 13),
        ("Fold Results", 3, 11, 1, 9),
        ("XGB by Liquid", 3, 14, 1, 8),
        ("XGB Predictions", 3, 11, 1, 11),
        ("Datasets", 3, 18, 1, 10),
        ("Configuration", 10, 16, 1, 4),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), facecolor="white")
    for ax, (sheet_name, min_row, max_row, min_col, max_col) in zip(axes.ravel(), previews):
        table_preview(
            ax,
            wb[sheet_name],
            sheet_name,
            min_row,
            max_row,
            min_col,
            max_col,
            5.5 if max_col > 9 else 6.5,
        )
    fig.tight_layout(pad=2.0)
    fig.savefig(PREVIEW_DIR / "all_sheets_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(PREVIEW_DIR))


if __name__ == "__main__":
    main()
