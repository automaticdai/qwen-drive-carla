param(
    [Parameter(Mandatory=$true)][string]$CarlaRoot,
    [int]$Port = 2000,
    [ValidateSet("Low", "Epic")][string]$Quality = "Epic",
    [switch]$Visible,
    [int]$Width = 1920,
    [int]$Height = 1080,
    [string]$LogDirectory = ""
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $CarlaRoot "CarlaUE4.exe"
if (-not (Test-Path $exe)) {
    throw "CarlaUE4.exe not found. Extract the official CARLA 0.9.16 Windows package and pass its root directory."
}
# Off-screen rendering still produces RGB camera images.
$arguments = @("-quality-level=$Quality", "-carla-rpc-port=$Port", "-nosound", "-stdout", "-FullStdOutLogOutput")
if ($Visible) {
    $arguments += @("-windowed", "-ForceRes", "-ResX=$Width", "-ResY=$Height")
} else {
    $arguments += "-RenderOffScreen"
}
$launch = @{
    FilePath = $exe
    WorkingDirectory = $CarlaRoot
    ArgumentList = $arguments
    PassThru = $true
}
if ($LogDirectory) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $launch.RedirectStandardOutput = Join-Path $LogDirectory "stdout.log"
    $launch.RedirectStandardError = Join-Path $LogDirectory "stderr.log"
}
# Apply DPI awareness only to this process tree, without changing desktop scaling
# or permanent Windows compatibility settings.
$previousCompatibility = $env:__COMPAT_LAYER
try {
    if ($Visible) { $env:__COMPAT_LAYER = "$previousCompatibility HIGHDPIAWARE".Trim() }
    $process = Start-Process @launch
} finally {
    $env:__COMPAT_LAYER = $previousCompatibility
}
Write-Output "CARLA launcher PID: $($process.Id)"
