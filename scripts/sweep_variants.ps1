# Build each rasterizer variant, benchmark it and compare its gradients to the upstream dump.
# Usage: pwsh scripts\sweep_variants.ps1 [-Model outputs\baseline\train] [-Scene train]
param([string]$Model = "outputs\baseline\train", [string]$Scene = "train")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
$variants = [ordered]@{
  "opt_off"     = ""
  "shmem_feat"  = "OPT_SHMEM_FEATURES=1"
  "warp_reduce" = "OPT_WARP_REDUCE_GRADS=1"
  "both"        = "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1"
  "sep_sh"      = "OPT_SEPARATE_SH=1"
  "all3"        = "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1 OPT_SEPARATE_SH=1"
  "tile_order"  = "OPT_TILE_ORDER=1"
  "all4"        = "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1 OPT_SEPARATE_SH=1 OPT_TILE_ORDER=1"
}
foreach ($name in $variants.Keys) {
  Write-Host "########## $name [$($variants[$name])]"
  cmd /c "scripts\build_rasterizer.bat opt `"$($variants[$name])`" > outputs\build_$name.log 2>&1"
  if ($LASTEXITCODE -ne 0) { throw "build failed for $name, see outputs\build_$name.log" }
  & $py p1-3dgs\bench.py -m $Model --label $name --json "outputs\bench\${Scene}_$name.json" | Out-Null
  & $py p1-3dgs\dump_grads.py -m $Model --out "outputs\grads\$name.pt" | Out-Null
  & $py p1-3dgs\dump_grads.py --compare outputs\grads\upstream.pt "outputs\grads\$name.pt" | Select-Object -Last 1
}
# leave the upstream build installed
cmd /c "scripts\build_rasterizer.bat upstream > outputs\build_upstream.log 2>&1"
& $py p1-3dgs\summarize.py outputs\bench\${Scene}_*.json
