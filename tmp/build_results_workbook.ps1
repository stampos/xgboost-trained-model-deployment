param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$builder = Join-Path $PSScriptRoot "build_results_workbook.py"
& $Python $builder
if ($LASTEXITCODE -ne 0) {
    throw "Workbook generation failed with exit code $LASTEXITCODE"
}
