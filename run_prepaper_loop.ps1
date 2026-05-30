param(
  [Parameter(Mandatory=$true)][string]$PrePaperMonday
)

$ErrorActionPreference = "Stop"

function Invoke-Py([string]$Cmd) {
  Write-Host "==> $Cmd"
  Invoke-Expression $Cmd
  if ($LASTEXITCODE -ne 0) { throw "FAILED (exit=$LASTEXITCODE): $Cmd" }
}

# ---- Stage 1A/1B (interactive) ----
Invoke-Py "`"$PrePaperMonday`" | python .\Optimizer_Stage1A_1B_v29R_CLEAN.py"
Write-Host "Stage1A_1B COMPLETED"

# ---- Stage 2 (interactive) ----
Invoke-Py "`"$PrePaperMonday`" | python .\Optimizer_Stage2_v29R_DualTF_CLEAN.py"
Write-Host "Stage2 COMPLETED - PLEASE REVIEW THE stage2_intraday_dual_tf_improved.csv"

# ---- Stage 4B (automatic) ----
Invoke-Py "python .\Optimizer_Stage4B_Master_Optimizer.py --prepaper-start `"$PrePaperMonday`""
Write-Host "Stage4B COMPLETED - Darwinian survivors generated"

# ---- Load Stage4B finalists (fallback to Stage2) ----
$stage4bPath = "results_v29R_30d\stage4b_intraday_dual_tf_selected.csv"
$stage2Path  = "results_v29R_30d\stage2_intraday_dual_tf_improved.csv"

if (Test-Path $stage4bPath) {
  $finalistsPath = $stage4bPath
  Write-Host "Using Stage4B pruned finalists: $finalistsPath"
}
elseif (Test-Path $stage2Path) {
  $finalistsPath = $stage2Path
  Write-Host "Stage4B finalists not found. Falling back to Stage2: $finalistsPath"
}
else {
  throw "Neither Stage4B nor Stage2 CSV found. Missing: $stage4bPath and $stage2Path"
}

$rows = Import-Csv $finalistsPath
if ($rows.Count -eq 0) { throw "Finalists CSV is empty: $finalistsPath" }

# Stage2/4B csv uses symbol (you confirmed)
$symbolCol = (("symbol","Symbol") | Where-Object { $_ -in $rows[0].PSObject.Properties.Name } | Select-Object -First 1)
if (-not $symbolCol) {
  throw "Finalists CSV missing 'symbol' column. Columns: $($rows[0].PSObject.Properties.Name -join ', ')"
}

$maxShow = [Math]::Min(50, $rows.Count)

# ---- Main loop ----
while ($true) {
  Write-Host ""
  Write-Host "Select a pair (top $maxShow shown). Enter Q to quit:"
  for ($i=0; $i -lt $maxShow; $i++) {
    $p = ($rows[$i].$symbolCol).ToString().Trim().ToUpper()
    "{0,2}) {1}" -f ($i+1), $p | Write-Host
  }
  Write-Host ""

  $choice = Read-Host "Enter number (1-$maxShow) or Q"
  if ($choice.ToUpper() -eq "Q") { break }

  if (-not ($choice -as [int])) { Write-Host "Invalid selection: $choice"; continue }
  $idx = [int]$choice - 1
  if ($idx -lt 0 -or $idx -ge $maxShow) { Write-Host "Out of range: $choice"; continue }

  $Pair = ($rows[$idx].$symbolCol).ToString().Trim().ToUpper()
  Write-Host "Selected pair: $Pair"
  Write-Host ""

  try {
    # Funnel -> Eventstudy -> Derive
    Invoke-Py "python .\Funnel_Data_Test_V30_EventStudy.py --pair $Pair --prepaper-start $PrePaperMonday"
    Write-Host "Funnel COMPLETED"

    Invoke-Py "python .\eventstudy_analysis.py --pair $Pair --prepaper-start $PrePaperMonday"
    Write-Host "eventstudy COMPLETED"

    Invoke-Py "python .\Derive_k_t_from_PQ_windows.py --pair $Pair --prepaper-start $PrePaperMonday"
    Write-Host "Derive COMPLETED"

    # Copy JSON
    $candSrc = ".\candidate_exit_params_${Pair}_prepaper_${PrePaperMonday}.json"
    $candDst = ".\candidate_for_TRADE.json"
    if (-not (Test-Path $candSrc)) { throw "Candidate JSON not found: $candSrc" }
    Copy-Item -Force $candSrc $candDst
    Write-Host "candidate_for_TRADE.json UPDATED from: $candSrc"

    # 7-day (Bypass interactive prompts and pass parameters automatically)
    Write-Host "Starting 7-day now (automated mode)."
    Invoke-Py "python '.\7_day_trade_window_forward_livefetch_v6+PrePaper.py' --pair $Pair --prepaper-start $PrePaperMonday --mode BACKTEST --verbose N"

    Write-Host "7-day COMPLETED for $Pair"
  }
  catch {
    Write-Host ""
    Write-Host "PAIR FAILED: $Pair"
    Write-Host $_.Exception.Message
    Write-Host "Continuing to next selection..."
    continue
  }
}

Write-Host "Done."