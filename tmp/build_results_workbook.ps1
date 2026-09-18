param(
    [string]$ResultsJson = ".\tmp\ml_comparison_results.json",
    [string]$OutputPath = ".\outputs\ml-comparison-20260918\ml_algorithm_results.xlsx",
    [string]$QaDirectory = ".\tmp\workbook_qa"
)

$ErrorActionPreference = "Stop"

function Get-ExcelColor([string]$Hex) {
    $clean = $Hex.TrimStart('#')
    $r = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return $r + (256 * $g) + (65536 * $b)
}

function Set-RangeData($Sheet, [int]$StartRow, [int]$StartColumn, [object[]]$Rows) {
    if ($Rows.Count -eq 0) { return $null }
    $columnCount = $Rows[0].Count
    $matrix = New-Object 'object[,]' $Rows.Count, $columnCount
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        for ($j = 0; $j -lt $columnCount; $j++) {
            $matrix[$i, $j] = $Rows[$i][$j]
        }
    }
    $target = $Sheet.Range(
        $Sheet.Cells.Item($StartRow, $StartColumn),
        $Sheet.Cells.Item($StartRow + $Rows.Count - 1, $StartColumn + $columnCount - 1)
    )
    $target.Value2 = $matrix
    return $target
}

function Format-Title($Sheet, [string]$RangeAddress, [string]$Text) {
    $range = $Sheet.Range($RangeAddress)
    $range.Merge()
    $range.Value2 = $Text
    $range.Interior.Color = Get-ExcelColor "1F4E78"
    $range.Font.Color = Get-ExcelColor "FFFFFF"
    $range.Font.Bold = $true
    $range.Font.Size = 18
    $range.HorizontalAlignment = -4108
    $range.VerticalAlignment = -4108
    $range.RowHeight = 32
}

function Format-SectionHeader($Range) {
    $Range.Interior.Color = Get-ExcelColor "D9EAF7"
    $Range.Font.Color = Get-ExcelColor "1F1F1F"
    $Range.Font.Bold = $true
    $Range.Borders.Item(9).Color = Get-ExcelColor "9EADBA"
    $Range.Borders.Item(9).Weight = 2
}

function Add-ExcelTable($Sheet, $Range, [string]$Name, [string]$Style = "TableStyleMedium2") {
    $table = $Sheet.ListObjects.Add(1, $Range, $null, 1)
    $table.Name = $Name
    $table.TableStyle = $Style
    return $table
}

function Finalize-Sheet($Sheet, [int]$FreezeRow = 0, [int]$MaxWidth = 28) {
    $used = $Sheet.UsedRange
    $used.Font.Name = "Aptos"
    $used.Font.Size = 10
    $used.VerticalAlignment = -4108
    $used.Columns.AutoFit() | Out-Null
    for ($col = 1; $col -le $used.Columns.Count; $col++) {
        $column = $Sheet.Columns.Item($col)
        if ($column.ColumnWidth -gt $MaxWidth) { $column.ColumnWidth = $MaxWidth }
    }
    if ($FreezeRow -gt 0) {
        $Sheet.Activate()
        $script:excel.ActiveWindow.FreezePanes = $false
        $script:excel.ActiveWindow.SplitColumn = 0
        $script:excel.ActiveWindow.SplitRow = $FreezeRow
        $script:excel.ActiveWindow.FreezePanes = $true
    }
    $Sheet.PageSetup.Orientation = 2
    $Sheet.PageSetup.Zoom = $false
    $Sheet.PageSetup.FitToPagesWide = 1
    $Sheet.PageSetup.FitToPagesTall = $false
    $Sheet.PageSetup.LeftMargin = $script:excel.InchesToPoints(0.25)
    $Sheet.PageSetup.RightMargin = $script:excel.InchesToPoints(0.25)
    $Sheet.PageSetup.TopMargin = $script:excel.InchesToPoints(0.4)
    $Sheet.PageSetup.BottomMargin = $script:excel.InchesToPoints(0.4)
}

$jsonPath = (Resolve-Path -LiteralPath $ResultsJson).Path
$result = Get-Content -Raw -Encoding UTF8 -LiteralPath $jsonPath | ConvertFrom-Json
$outputFullPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputPath))
$qaFullPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $QaDirectory))
$outputFolder = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputFolder | Out-Null
New-Item -ItemType Directory -Force -Path $qaFullPath | Out-Null

$excel = $null
$workbook = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $script:excel = $excel
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Add()

    while ($workbook.Worksheets.Count -lt 7) { $workbook.Worksheets.Add() | Out-Null }
    $summarySheet = $workbook.Worksheets.Item(1); $summarySheet.Name = "Summary"
    $avgSheet = $workbook.Worksheets.Item(2); $avgSheet.Name = "Average Results"
    $foldSheet = $workbook.Worksheets.Item(3); $foldSheet.Name = "Fold Results"
    $liquidSheet = $workbook.Worksheets.Item(4); $liquidSheet.Name = "XGB by Liquid"
    $predictionSheet = $workbook.Worksheets.Item(5); $predictionSheet.Name = "XGB Predictions"
    $datasetSheet = $workbook.Worksheets.Item(6); $datasetSheet.Name = "Datasets"
    $configSheet = $workbook.Worksheets.Item(7); $configSheet.Name = "Configuration"

    foreach ($sheet in @($summarySheet, $avgSheet, $foldSheet, $liquidSheet, $predictionSheet, $datasetSheet, $configSheet)) {
        $sheet.Cells.Clear() | Out-Null
        $sheet.DisplayPageBreaks = $false
    }

    # Summary sheet
    Format-Title $summarySheet "A1:M1" "Regression Algorithm Comparison"
    $summarySheet.Range("A3:M3").Merge()
    $summarySheet.Range("A3").Value2 = "5-fold cross-validation | KFold(shuffle=True, random_state=42) | Lower MAE/RMSE and higher R2 are better"
    $summarySheet.Range("A3:M3").Interior.Color = Get-ExcelColor "F2F2F2"
    $summarySheet.Range("A3:M3").Font.Italic = $true
    $summarySheet.Range("A3:M3").HorizontalAlignment = -4108

    $datasetOne = $result.datasets | Where-Object dataset -eq "mesos_oros.csv"
    $datasetTwo = $result.datasets | Where-Object dataset -eq "mesos_oros_nomeli.csv"
    $cardOne = @(
        @("Dataset", $datasetOne.dataset, "Rows", [int]$datasetOne.rows, "Liquids", [int]$datasetOne.liquids, "Missing", [int]$datasetOne.missing_values),
        @("Target mean", [double]$datasetOne.target_mean, "Target std", [double]$datasetOne.target_std, "Min", [double]$datasetOne.target_min, "Max", [double]$datasetOne.target_max)
    )
    $cardTwo = @(
        @("Dataset", $datasetTwo.dataset, "Rows", [int]$datasetTwo.rows, "Liquids", [int]$datasetTwo.liquids, "Missing", [int]$datasetTwo.missing_values),
        @("Target mean", [double]$datasetTwo.target_mean, "Target std", [double]$datasetTwo.target_std, "Min", [double]$datasetTwo.target_min, "Max", [double]$datasetTwo.target_max)
    )
    Set-RangeData $summarySheet 5 1 $cardOne | Out-Null
    Set-RangeData $summarySheet 5 7 $cardTwo | Out-Null
    $summarySheet.Range("A5:H6").Borders.LineStyle = 1
    $summarySheet.Range("G5:N6").Borders.LineStyle = 1
    $summarySheet.Range("A5:A6,C5:C6,E5:E6,G5:G6").Font.Bold = $true
    $summarySheet.Range("G5:G6,I5:I6,K5:K6,M5:M6").Font.Bold = $true
    $summarySheet.Range("A5:H6").Interior.Color = Get-ExcelColor "EAF2F8"
    $summarySheet.Range("G5:N6").Interior.Color = Get-ExcelColor "FCE4D6"
    $summarySheet.Range("B6,D6,F6,H6,H6,J6,L6,N6").NumberFormat = "0.0000"

    $headers = @("Rank", "Algorithm", "MAE", "RMSE", "R2", "Elapsed (s)")
    $rankOne = @($result.summary_results | Where-Object dataset -eq "mesos_oros.csv" | Sort-Object mae_mean)
    $rankTwo = @($result.summary_results | Where-Object dataset -eq "mesos_oros_nomeli.csv" | Sort-Object mae_mean)
    $tableOneRows = New-Object System.Collections.Generic.List[object]
    $tableTwoRows = New-Object System.Collections.Generic.List[object]
    $tableOneRows.Add($headers)
    $tableTwoRows.Add($headers)
    for ($i = 0; $i -lt $rankOne.Count; $i++) {
        $r = $rankOne[$i]
        $tableOneRows.Add(@($i + 1, $r.algorithm, [double]$r.mae_mean, [double]$r.rmse_mean, [double]$r.r2_mean, [double]$r.total_elapsed_s))
    }
    for ($i = 0; $i -lt $rankTwo.Count; $i++) {
        $r = $rankTwo[$i]
        $tableTwoRows.Add(@($i + 1, $r.algorithm, [double]$r.mae_mean, [double]$r.rmse_mean, [double]$r.r2_mean, [double]$r.total_elapsed_s))
    }
    $summarySheet.Range("A9:F9").Merge(); $summarySheet.Range("A9").Value2 = "mesos_oros.csv - ranked by MAE"
    $summarySheet.Range("H9:M9").Merge(); $summarySheet.Range("H9").Value2 = "mesos_oros_nomeli.csv - ranked by MAE"
    Format-SectionHeader $summarySheet.Range("A9:F9")
    Format-SectionHeader $summarySheet.Range("H9:M9")
    $rangeOne = Set-RangeData $summarySheet 10 1 $tableOneRows.ToArray()
    $rangeTwo = Set-RangeData $summarySheet 10 8 $tableTwoRows.ToArray()
    Add-ExcelTable $summarySheet $rangeOne "SummaryRankFull" "TableStyleMedium2" | Out-Null
    Add-ExcelTable $summarySheet $rangeTwo "SummaryRankNoMeli" "TableStyleMedium9" | Out-Null
    $summarySheet.Range("C11:F15").NumberFormat = "0.0000"
    $summarySheet.Range("J11:M15").NumberFormat = "0.0000"
    $summarySheet.Range("A11:F11").Interior.Color = Get-ExcelColor "E2F0D9"
    $summarySheet.Range("H11:M11").Interior.Color = Get-ExcelColor "E2F0D9"
    $summarySheet.Range("A11:F11,H11:M11").Font.Bold = $true

    $chartRows = New-Object System.Collections.Generic.List[object]
    $chartRows.Add(@("Algorithm", "mesos_oros.csv", "mesos_oros_nomeli.csv"))
    foreach ($algorithm in @("Linear Regression", "Polynomial Ridge", "Random Forest", "XGBoost", "Neural Network")) {
        $one = $result.summary_results | Where-Object { $_.dataset -eq "mesos_oros.csv" -and $_.algorithm -eq $algorithm }
        $two = $result.summary_results | Where-Object { $_.dataset -eq "mesos_oros_nomeli.csv" -and $_.algorithm -eq $algorithm }
        $chartRows.Add(@($algorithm, [double]$one.r2_mean, [double]$two.r2_mean))
    }
    $chartRange = Set-RangeData $summarySheet 19 1 $chartRows.ToArray()
    Add-ExcelTable $summarySheet $chartRange "SummaryR2ChartData" "TableStyleMedium2" | Out-Null
    $summarySheet.Range("B20:C24").NumberFormat = "0.0000"
    $chartObject = $summarySheet.ChartObjects().Add(475, 330, 610, 300)
    $chart = $chartObject.Chart
    $chart.ChartType = 51
    $chart.SetSourceData($chartRange)
    $chart.HasTitle = $true
    $chart.ChartTitle.Text = "Mean R2 by algorithm"
    $chart.HasLegend = $true
    $chart.Legend.Position = -4107
    $chart.Axes(2).MinimumScale = -0.25
    $chart.Axes(2).MaximumScale = 1.0
    $chart.Axes(2).TickLabels.NumberFormat = "0.00"

    # Average results
    Format-Title $avgSheet "A1:M1" "Mean Metrics and Standard Deviations by Model"
    $avgHeaders = @("Dataset", "Rank MAE", "Algorithm", "MAE mean", "MAE std", "MSE mean", "MSE std", "RMSE mean", "RMSE std", "R2 mean", "R2 std", "Fit mean (s)", "Total elapsed (s)")
    $avgRows = New-Object System.Collections.Generic.List[object]
    $avgRows.Add($avgHeaders)
    foreach ($datasetName in @("mesos_oros.csv", "mesos_oros_nomeli.csv")) {
        $ranked = @($result.summary_results | Where-Object dataset -eq $datasetName | Sort-Object mae_mean)
        for ($i = 0; $i -lt $ranked.Count; $i++) {
            $r = $ranked[$i]
            $avgRows.Add(@($datasetName, $i + 1, $r.algorithm, [double]$r.mae_mean, [double]$r.mae_std, [double]$r.mse_mean, [double]$r.mse_std, [double]$r.rmse_mean, [double]$r.rmse_std, [double]$r.r2_mean, [double]$r.r2_std, [double]$r.fit_time_mean_s, [double]$r.total_elapsed_s))
        }
    }
    $avgRange = Set-RangeData $avgSheet 3 1 $avgRows.ToArray()
    Add-ExcelTable $avgSheet $avgRange "AverageResults" "TableStyleMedium2" | Out-Null
    $avgSheet.Range("D4:M13").NumberFormat = "0.0000"
    $avgSheet.Range("D4:D13").FormatConditions.AddColorScale(3) | Out-Null
    $avgSheet.Range("J4:J13").FormatConditions.AddColorScale(3) | Out-Null

    # Fold results
    Format-Title $foldSheet "A1:I1" "Detailed Results by Fold"
    $foldHeaders = @("Dataset", "Algorithm", "Fold", "MAE", "MSE", "RMSE", "R2", "Fit time (s)", "Score time (s)")
    $foldRows = New-Object System.Collections.Generic.List[object]
    $foldRows.Add($foldHeaders)
    foreach ($r in $result.fold_results) {
        $foldRows.Add(@($r.dataset, $r.algorithm, [int]$r.fold, [double]$r.mae, [double]$r.mse, [double]$r.rmse, [double]$r.r2, [double]$r.fit_time_s, [double]$r.score_time_s))
    }
    $foldRange = Set-RangeData $foldSheet 3 1 $foldRows.ToArray()
    Add-ExcelTable $foldSheet $foldRange "FoldResults" "TableStyleMedium2" | Out-Null
    $foldSheet.Range("D4:I53").NumberFormat = "0.0000"

    # XGBoost by liquid
    Format-Title $liquidSheet "A1:H1" "XGBoost: Out-of-Fold MAE by Liquid"
    $liquidHeaders = @("Dataset", "Liquid", "Count", "Actual mean (s)", "Prediction mean (s)", "MAE (s)", "Abs. error std (s)", "Overall MAE (s)")
    $liquidRows = New-Object System.Collections.Generic.List[object]
    $liquidRows.Add($liquidHeaders)
    foreach ($r in $result.xgboost_liquid_summary) {
        $liquidRows.Add(@($r.dataset, $r.liquid, [int]$r.count, [double]$r.actual_mean, [double]$r.prediction_mean, [double]$r.mae, [double]$r.error_std, [double]$r.overall_mae))
    }
    $liquidRange = Set-RangeData $liquidSheet 3 1 $liquidRows.ToArray()
    Add-ExcelTable $liquidSheet $liquidRange "XgbLiquidResults" "TableStyleMedium9" | Out-Null
    $liquidSheet.Range("D4:H14").NumberFormat = "0.0000"
    $liquidSheet.Range("F4:F14").FormatConditions.AddColorScale(3) | Out-Null

    # XGBoost OOF predictions
    Format-Title $predictionSheet "A1:K1" "XGBoost: Out-of-Fold Predictions"
    $predictionHeaders = @("Dataset", "Row", "Fold", "Liquid", "Height (cm)", "Shape", "Mass (g)", "Actual (s)", "Prediction (s)", "Error (s)", "Absolute error (s)")
    $predictionRows = New-Object System.Collections.Generic.List[object]
    $predictionRows.Add($predictionHeaders)
    foreach ($r in $result.xgboost_oof_predictions) {
        $predictionRows.Add(@($r.dataset, [int]$r.row, [int]$r.fold, $r.liquid, [double]$r.height_cm, $r.shape, [double]$r.mass_g, [double]$r.actual_s, [double]$r.prediction_s, [double]$r.error_s, [double]$r.absolute_error_s))
    }
    $predictionRange = Set-RangeData $predictionSheet 3 1 $predictionRows.ToArray()
    Add-ExcelTable $predictionSheet $predictionRange "XgbOofPredictions" "TableStyleMedium2" | Out-Null
    $predictionSheet.Range("E4:E393,G4:G393").NumberFormat = "0.00"
    $predictionSheet.Range("H4:K393").NumberFormat = "0.0000"
    $predictionSheet.Range("K4:K393").FormatConditions.AddColorScale(3) | Out-Null

    # Dataset profile
    Format-Title $datasetSheet "A1:J1" "Dataset Profiles"
    $datasetHeaders = @("Dataset", "Rows", "Columns", "Missing", "Liquids", "Shapes", "Target mean", "Target std", "Target min", "Target max")
    $datasetRows = New-Object System.Collections.Generic.List[object]
    $datasetRows.Add($datasetHeaders)
    foreach ($d in $result.datasets) {
        $datasetRows.Add(@($d.dataset, [int]$d.rows, [int]$d.columns, [int]$d.missing_values, [int]$d.liquids, [int]$d.shapes, [double]$d.target_mean, [double]$d.target_std, [double]$d.target_min, [double]$d.target_max))
    }
    $datasetRange = Set-RangeData $datasetSheet 3 1 $datasetRows.ToArray()
    Add-ExcelTable $datasetSheet $datasetRange "DatasetProfiles" "TableStyleMedium2" | Out-Null
    $datasetSheet.Range("G4:J5").NumberFormat = "0.0000"
    $datasetSheet.Range("A8:C8").Value2 = @("Dataset", "Liquid", "Count")
    $countRows = New-Object System.Collections.Generic.List[object]
    foreach ($d in $result.datasets) {
        foreach ($property in $d.liquid_counts.PSObject.Properties) {
            $countRows.Add(@($d.dataset, $property.Name, [int]$property.Value))
        }
    }
    $countData = Set-RangeData $datasetSheet 9 1 $countRows.ToArray()
    $countRange = $datasetSheet.Range("A8", $datasetSheet.Cells.Item(8 + $countRows.Count, 3))
    Add-ExcelTable $datasetSheet $countRange "DatasetLiquidCounts" "TableStyleMedium9" | Out-Null

    # Configuration and provenance
    Format-Title $configSheet "A1:D1" "Experiment Configuration and Provenance"
    $configSheet.Range("A3:B7").Value2 = @(
        @("Method", $result.cv.method),
        @("Folds", [int]$result.cv.n_splits),
        @("Shuffle", [bool]$result.cv.shuffle),
        @("Random state", [int]$result.cv.random_state),
        @("Target", $result.target)
    )
    $configSheet.Range("A3:A7").Font.Bold = $true
    $configSheet.Range("A3:B7").Interior.Color = Get-ExcelColor "F2F2F2"
    $configHeaders = @("Algorithm / analysis", "Source file", "Parameters", "Note")
    $configRows = New-Object System.Collections.Generic.List[object]
    $configRows.Add($configHeaders)
    foreach ($c in $result.model_configs) {
        $note = if ($c.algorithm -eq "XGBoost MAE per liquid") { "Additional analysis of the same XGBoost model; not a separate ranked model" } else { "5-fold cross-validation" }
        $configRows.Add(@($c.algorithm, $c.source_file, $c.parameters, $note))
    }
    $configRange = Set-RangeData $configSheet 10 1 $configRows.ToArray()
    Add-ExcelTable $configSheet $configRange "ExperimentConfiguration" "TableStyleMedium2" | Out-Null
    $configSheet.Range("A18:B20").Value2 = @(
        @("Generated UTC", $result.generated_at_utc),
        @("Warnings", [int]$result.warnings.Count),
        @("Note", "Metrics come from the actual model executions.")
    )
    $configSheet.Range("A18:A20").Font.Bold = $true

    Finalize-Sheet $summarySheet 3 24
    Finalize-Sheet $avgSheet 3 24
    Finalize-Sheet $foldSheet 3 24
    Finalize-Sheet $liquidSheet 3 24
    Finalize-Sheet $predictionSheet 3 24
    Finalize-Sheet $datasetSheet 3 24
    Finalize-Sheet $configSheet 3 42
    $summarySheet.Columns.Item(2).ColumnWidth = 21
    $summarySheet.Columns.Item(9).ColumnWidth = 21
    $configSheet.Columns.Item(2).ColumnWidth = 32
    $configSheet.Columns.Item(3).ColumnWidth = 70
    $configSheet.Columns.Item(4).ColumnWidth = 48
    $configSheet.Range("C11:D16").WrapText = $true
    $configSheet.Range("C11:D16").Rows.AutoFit() | Out-Null
    $predictionSheet.Columns.Item(1).ColumnWidth = 24
    $foldSheet.Columns.Item(1).ColumnWidth = 24
    $avgSheet.Columns.Item(1).ColumnWidth = 24
    $liquidSheet.Columns.Item(1).ColumnWidth = 24

    $summarySheet.Activate()
    $excel.ActiveWindow.DisplayGridlines = $false
    foreach ($sheet in @($avgSheet, $foldSheet, $liquidSheet, $predictionSheet, $datasetSheet, $configSheet)) {
        $sheet.Activate()
        $excel.ActiveWindow.DisplayGridlines = $false
    }
    $summarySheet.Activate()

    $workbook.SaveAs($outputFullPath, 51)

    foreach ($sheet in @($summarySheet, $avgSheet, $foldSheet, $liquidSheet, $predictionSheet, $datasetSheet, $configSheet)) {
        $safeName = ($sheet.Name -replace '[^\p{L}\p{Nd}_-]', '_')
        $pdfPath = Join-Path $qaFullPath ($safeName + ".pdf")
        $sheet.ExportAsFixedFormat(0, $pdfPath)
    }

    $verification = [pscustomobject]@{
        Workbook = $outputFullPath
        Sheets = $workbook.Worksheets.Count
        SummaryRows = $result.summary_results.Count
        FoldRows = $result.fold_results.Count
        LiquidRows = $result.xgboost_liquid_summary.Count
        PredictionRows = $result.xgboost_oof_predictions.Count
        Warnings = $result.warnings.Count
        BestFull = ($rankOne[0].algorithm)
        BestNoMeli = ($rankTwo[0].algorithm)
        SummaryCell = $summarySheet.Range("B11").Value2
    }
    $verification | ConvertTo-Json -Compress
}
finally {
    if ($workbook -ne $null) { $workbook.Close($true) }
    if ($excel -ne $null) { $excel.Quit() }
    foreach ($item in @($chart, $chartObject, $configSheet, $datasetSheet, $predictionSheet, $liquidSheet, $foldSheet, $avgSheet, $summarySheet, $workbook, $excel)) {
        if ($null -ne $item) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($item) }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
