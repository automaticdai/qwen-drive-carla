param([int]$Width = 1920, [int]$Height = 1080)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class DenseDebugLayout {
 [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out Rect r);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out Rect r);
 [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int command);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int height, uint flags);
}
'@
[DenseDebugLayout]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName System.Windows.Forms
$area = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$carla = Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $carla) { throw 'No visible CARLA window found. Start CARLA with -Visible first.' }
[DenseDebugLayout]::ShowWindowAsync($carla.MainWindowHandle, 9) | Out-Null
Start-Sleep -Milliseconds 300
$client = New-Object DenseDebugLayout+Rect
$outer = New-Object DenseDebugLayout+Rect
[DenseDebugLayout]::GetClientRect($carla.MainWindowHandle, [ref]$client) | Out-Null
[DenseDebugLayout]::GetWindowRect($carla.MainWindowHandle, [ref]$outer) | Out-Null
$outerWidth = $Width + ($outer.Right - $outer.Left) - ($client.Right - $client.Left)
$outerHeight = $Height + ($outer.Bottom - $outer.Top) - ($client.Bottom - $client.Top)
$browserWidth = $area.Width - $outerWidth
if ($browserWidth -lt 400 -or $outerHeight -gt $area.Height) { throw 'Desktop is too small for this CARLA resolution beside a usable dashboard.' }
# A paused synchronous CARLA session may need one dashboard Step to process placement.
[DenseDebugLayout]::SetWindowPos($carla.MainWindowHandle, [IntPtr]::Zero, $area.X, $area.Y, $outerWidth, $outerHeight, 0x0060) | Out-Null
[DenseDebugLayout]::SetForegroundWindow($carla.MainWindowHandle) | Out-Null
$browser = Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '*dense traffic debug*' -and $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $browser) {
 Start-Process 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' -ArgumentList '--app=http://localhost:8877','--new-window'
 for ($attempt = 0; $attempt -lt 15 -and -not $browser; $attempt++) {
  Start-Sleep -Seconds 1
  $browser = Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '*dense traffic debug*' -and $_.MainWindowHandle -ne 0 } | Select-Object -First 1
 }
}
if (-not $browser) { throw 'Could not find the dense traffic debug browser window.' }
[DenseDebugLayout]::ShowWindowAsync($browser.MainWindowHandle, 9) | Out-Null
Start-Sleep -Milliseconds 300
[DenseDebugLayout]::SetWindowPos($browser.MainWindowHandle, [IntPtr]::Zero, ($area.X + $outerWidth), $area.Y, $browserWidth, $area.Height, 0x0060) | Out-Null
[DenseDebugLayout]::SetForegroundWindow($browser.MainWindowHandle) | Out-Null
Write-Output "Requested CARLA client ${Width}x${Height}, dashboard width $browserWidth; side by side."
