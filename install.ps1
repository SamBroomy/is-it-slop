#!/usr/bin/env pwsh
# is-it-slop Windows installer (PowerShell)
# https://github.com/SamBroomy/is-it-slop
#
# Quick install:
#   Invoke-RestMethod https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.ps1 | Invoke-Expression
#
# Install specific version:
#   $env:ISITSLOP_VERSION='v0.6.0'; Invoke-RestMethod https://raw.githubusercontent.com/SamBroomy/is-it-slop/main/install.ps1 | Invoke-Expression

param(
    [string]$Version = if ($env:ISITSLOP_VERSION) { $env:ISITSLOP_VERSION } else { "latest" }
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Repo = "SamBroomy/is-it-slop"
$BinaryName = "is-it-slop"
$BinaryExe = "${BinaryName}.exe"
$InstallDir = if ($env:INSTALL_DIR) { $env:INSTALL_DIR } else { "$HOME\.local\bin" }

function Write-Info([string]$Msg) {
    Write-Host $Msg -ForegroundColor Green
}

function Write-Warn([string]$Msg) {
    Write-Host "Warning: $Msg" -ForegroundColor Yellow
}

function Write-Error-Msg([string]$Msg) {
    Write-Host "Error: $Msg" -ForegroundColor Red
}

function Get-Architecture {
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "32-bit systems are not supported"
    }
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
        return "aarch64"
    }
    return "x86_64"
}

function Get-LatestVersion {
    Write-Info "Fetching latest release information..."
    $apiUrl = "https://api.github.com/repos/$Repo/releases/latest"
    $headers = @{}
    if ($env:GITHUB_TOKEN) {
        $headers["Authorization"] = "Bearer $env:GITHUB_TOKEN"
    }
    try {
        $release = Invoke-RestMethod -Uri $apiUrl -Method Get -Headers $headers
        if (-not $release.tag_name) {
            throw "No version found in release"
        }
        return $release.tag_name
    } catch {
        throw "Could not determine latest version: $_`n`nThis may be due to GitHub API rate limiting. Try:`n  1. Wait a minute and try again`n  2. Set `$env:GITHUB_TOKEN to use an authenticated request`n  3. Specify a version explicitly:`n     `$env:ISITSLOP_VERSION='v0.6.3'; Invoke-RestMethod https://... | Invoke-Expression"
    }
}

function Test-ExistingInstallation {
    $binPath = Join-Path $InstallDir $BinaryExe
    if (Test-Path $binPath) {
        try {
            $currentVersion = & $binPath --version 2>$null
            Write-Info "Existing installation detected: $currentVersion"
        } catch {
            Write-Info "Existing installation detected: unknown version"
        }
        Write-Info "Replacing existing installation..."
    }
}

function Install-Binary {
    param([string]$Version)

    $Script:TempDir = $null

    try {
        Write-Info "Installing $BinaryName"

        $arch = Get-Architecture
        $target = "${arch}-pc-windows-msvc"
        Write-Info "Detected platform: $target"

        if ($Version -eq "latest") {
            $Version = Get-LatestVersion
        }
        Write-Info "Version: $Version"

        Test-ExistingInstallation

        $archiveName = "${BinaryName}-${target}.zip"
        $downloadUrl = "https://github.com/$Repo/releases/download/$Version/$archiveName"

        Write-Info "Downloading from $downloadUrl"

        $Script:TempDir = Join-Path $env:TEMP "is-it-slop-$([System.Guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Path $Script:TempDir -Force | Out-Null
        $archivePath = Join-Path $Script:TempDir $archiveName

        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
        } catch {
            throw "Failed to download binary from $downloadUrl`n`nPlease check:`n  1. Version $Version exists: https://github.com/$Repo/releases`n  2. Binary is available for platform: $target`n  3. Your internet connection is working"
        }

        Write-Info "Extracting archive..."
        $extractDir = Join-Path $Script:TempDir "extracted"
        Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force

        $binaryRelPath = Get-ChildItem -Path $extractDir -Recurse -Name $BinaryExe | Select-Object -First 1
        if (-not $binaryRelPath) {
            throw "Could not find $BinaryExe in archive"
        }
        $binaryPath = Join-Path $extractDir $binaryRelPath

        if (-not (Test-Path $InstallDir)) {
            Write-Info "Creating installation directory: $InstallDir"
            New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        }

        Write-Info "Installing to $InstallDir"
        $destPath = Join-Path $InstallDir $BinaryExe
        Move-Item -Path $binaryPath -Destination $destPath -Force

        if (-not (Test-Path $destPath)) {
            throw "Installation failed: binary not found at $destPath"
        }

        Write-Host ""
        Write-Info "Successfully installed $BinaryName"
        Write-Host ""

        try {
            $installedVersion = & $destPath --version 2>$null
            Write-Info "Installed version: $installedVersion"
        } catch {
            Write-Warn "Could not verify installed version"
        }
        Write-Host ""

        $currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User") ?? ""
        if (($currentUserPath -split ';') -contains $InstallDir) {
            Write-Info "$InstallDir is in your User PATH"
            Write-Host ""
            Write-Info "Try it now: $BinaryName --help"
        } else {
            Write-Warn "$InstallDir is NOT in your User PATH"
            Write-Host ""
            Write-Info "To add to PATH, run this command:"
            Write-Host "  [Environment]::SetEnvironmentVariable('PATH', `$env:PATH + ';$InstallDir', 'User')"
            Write-Host ""
            Write-Info "Then restart your terminal, or use the full path:"
            Write-Host "  & '$destPath' --help"
        }
    } finally {
        if ($Script:TempDir -and (Test-Path $Script:TempDir)) {
            Remove-Item -Path $Script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Install-Binary -Version $Version
