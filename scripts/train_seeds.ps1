# Full 30k-iteration training, several seeds per rasterizer build, then a mean ± std table.
# Usage: pwsh scripts\train_seeds.ps1 [-Scene train] [-Seeds 0,1,2] [-Builds upstream,all3]
param([string]$Scene = "train", [int[]]$Seeds = @(0, 1, 2), [string[]]$Builds = @("upstream", "all3"))
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
$defines = @{
  "upstream" = $null
  "both"     = "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1"
  "all3"     = "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1 OPT_SEPARATE_SH=1"
}
$data = Get-ChildItem -Directory -Path data\tandt, data\db -Filter $Scene | Select-Object -First 1
if (-not $data) { throw "scene $Scene not found under data\tandt or data\db" }
foreach ($b in $Builds) {
  Write-Host "########## build $b"
  if ($b -eq "upstream") { cmd /c "scripts\build_rasterizer.bat upstream > outputs\build_upstream.log 2>&1" }
  else { cmd /c "scripts\build_rasterizer.bat opt `"$($defines[$b])`" > outputs\build_$b.log 2>&1" }
  if ($LASTEXITCODE -ne 0) { throw "build failed for $b" }
  foreach ($s in $Seeds) {
    $out = "outputs\seeds\${Scene}\${b}_s$s"
    if (Test-Path "$out\train_run.json") { Write-Host "skip $out (done)"; continue }
    Write-Host "########## $b seed $s -> $out"
    New-Item -ItemType Directory -Force $out | Out-Null
    & $py p1-3dgs\train_seed.py --seed $s --json "$out\train_run.json" -- `
        -s $data.FullName -m $out --eval --disable_viewer *> "$out.log"
    if ($LASTEXITCODE -ne 0) { throw "training failed: $b seed $s, see $out.log" }
    Get-Content "$out\train_run.json"
  }
}
& $py p1-3dgs\summarize_seeds.py "outputs\seeds\$Scene"
