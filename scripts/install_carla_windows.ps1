param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [string]$Destination = (Join-Path $env:LOCALAPPDATA "CARLA\0.9.16")
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Archive not found: $Archive. Run scripts/download_carla_windows.py first."
}
if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists: $Destination. This installer does not overwrite an installation."
}
$staging = "$Destination.installing"
if (Test-Path -LiteralPath $staging) {
    throw "An incomplete extraction exists: $staging. Inspect it before retrying."
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null
Write-Output "Extracting CARLA into $staging"
& tar.exe -xf $Archive -C $staging
if ($LASTEXITCODE -ne 0) {
    throw "Archive extraction failed with exit code $LASTEXITCODE; partial files remain in $staging"
}
if (-not (Test-Path (Join-Path $staging "CarlaUE4.exe"))) {
    throw "The archive did not contain CarlaUE4.exe at its root; inspect $staging"
}
Move-Item -LiteralPath $staging -Destination $Destination
Write-Output "CARLA installed: $Destination"
