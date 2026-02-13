$ErrorActionPreference = "Stop"

$Repo = if ($env:UNBG_INSTALL_REPO) { $env:UNBG_INSTALL_REPO } else { "unbgio/core-sdk" }
$Version = if ($env:UNBG_INSTALL_VERSION) { $env:UNBG_INSTALL_VERSION } else { "latest" }
$BinaryName = if ($env:UNBG_BINARY_NAME) { $env:UNBG_BINARY_NAME } else { "unbg.exe" }
$InstallDir = if ($env:UNBG_INSTALL_DIR) { $env:UNBG_INSTALL_DIR } else { Join-Path $HOME ".local\bin" }

function Get-TargetArch {
  $arch = $null
  try {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
  }
  catch {
    $arch = $null
  }
  if ([string]::IsNullOrWhiteSpace($arch)) {
    $arch = $env:PROCESSOR_ARCHITECTURE
  }
  if ([string]::IsNullOrWhiteSpace($arch)) {
    throw "Could not detect CPU architecture."
  }
  $arch = $arch.ToLowerInvariant()
  switch ($arch) {
    "x64" { return "x86_64" }
    "amd64" { return "x86_64" }
    "arm64" { return "aarch64" }
    default { throw "Unsupported CPU architecture: $arch" }
  }
}

function Get-ReleaseAssetUrl([string]$Repo, [string]$Version, [string]$Target) {
  if ($Version -eq "latest") {
    return "https://github.com/$Repo/releases/latest/download/unbg-$Target.zip"
  }
  return "https://github.com/$Repo/releases/download/$Version/unbg-$Target.zip"
}

function Expand-ReleaseArchive([string]$ArchivePath, [string]$DestinationPath) {
  if ($ArchivePath.ToLowerInvariant().EndsWith(".zip")) {
    Expand-Archive -Path $ArchivePath -DestinationPath $DestinationPath -Force
    return
  }

  if ($ArchivePath.ToLowerInvariant().EndsWith(".tar.gz") -or $ArchivePath.ToLowerInvariant().EndsWith(".tgz")) {
    tar -xzf $ArchivePath -C $DestinationPath
    return
  }

  throw "Unsupported archive format: $ArchivePath"
}

function Ensure-UserPathContains([string]$Dir) {
  $normalizedDir = $Dir.Trim().TrimEnd("\")
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  $parts = @()
  if (-not [string]::IsNullOrWhiteSpace($userPath)) {
    $parts = $userPath -split ";" | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  }
  $alreadyExists = $false
  foreach ($part in $parts) {
    if ($part.TrimEnd("\").Equals($normalizedDir, [System.StringComparison]::OrdinalIgnoreCase)) {
      $alreadyExists = $true
      break
    }
  }
  if (-not $alreadyExists) {
    $newParts = @($parts + $Dir)
    $newUserPath = ($newParts -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    Write-Host "Added to user PATH: $Dir"
  } else {
    Write-Host "User PATH already contains: $Dir"
  }
  if (($env:PATH -split ";") -notcontains $Dir) {
    $env:PATH = "$env:PATH;$Dir"
  }
}

function Ensure-UserOrtPath([string]$InstallDir) {
  $dllPath = Join-Path $InstallDir "onnxruntime.dll"
  if (-not (Test-Path $dllPath)) {
    return
  }
  $current = [Environment]::GetEnvironmentVariable("ORT_DYLIB_PATH", "User")
  if ([string]::IsNullOrWhiteSpace($current) -or -not (Test-Path $current)) {
    [Environment]::SetEnvironmentVariable("ORT_DYLIB_PATH", $dllPath, "User")
    Write-Host "Set user ORT_DYLIB_PATH: $dllPath"
  }
  $env:ORT_DYLIB_PATH = $dllPath
}

$arch = Get-TargetArch
$target = "windows-$arch"
$assetUrl = Get-ReleaseAssetUrl -Repo $Repo -Version $Version -Target $target

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unbg-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmpRoot | Out-Null
$extractDir = Join-Path $tmpRoot "extract"
New-Item -ItemType Directory -Path $extractDir | Out-Null

try {
  $archiveName = "unbg-$target.zip"
  $archivePath = Join-Path $tmpRoot $archiveName
  Write-Host "Downloading $assetUrl"
  try {
    Invoke-WebRequest -Uri $assetUrl -OutFile $archivePath
  }
  catch {
    throw "Could not download asset for target $target in $Repo ($Version). Expected URL: $assetUrl"
  }

  Write-Host "Extracting $archiveName"
  Expand-ReleaseArchive -ArchivePath $archivePath -DestinationPath $extractDir

  $archiveFiles = Get-ChildItem -Path $extractDir -Recurse -File
  $binary = $archiveFiles | Where-Object { $_.Name -ieq $BinaryName } | Select-Object -First 1
  if (-not $binary) {
    throw "Could not find '$BinaryName' inside the release archive."
  }

  New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
  $destination = Join-Path $InstallDir $BinaryName
  foreach ($file in $archiveFiles) {
    Copy-Item -Path $file.FullName -Destination (Join-Path $InstallDir $file.Name) -Force
  }

  Write-Host "Installed $BinaryName to $destination"
  Ensure-UserPathContains -Dir $InstallDir
  Ensure-UserOrtPath -InstallDir $InstallDir
  Write-Host "Open a new terminal to use 'unbg' globally."
}
finally {
  Remove-Item -Path $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}
