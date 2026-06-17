param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Push-Location $Root

try {
  if (Get-Command gitleaks -ErrorAction SilentlyContinue) {
    gitleaks detect --source .
    if ($LASTEXITCODE -ne 0) { throw "gitleaks failed" }
  }

  $patterns = @(
    'C:\\Users\\ogosh',
    '/Users/s30519',
    '100\.\d+\.\d+\.\d+',
    'sk-[A-Za-z0-9_-]{20,}',
    'ghp_[A-Za-z0-9_]{20,}',
    'BEGIN (RSA |OPENSSH )?PRIVATE KEY'
  )

  $files = git ls-files | Where-Object { $_ -ne 'scripts/publish-preflight.ps1' }
  foreach ($pattern in $patterns) {
    $hits = $files | ForEach-Object {
      if (Test-Path -LiteralPath $_) {
        Select-String -Path $_ -Pattern $pattern -ErrorAction SilentlyContinue
      }
    }
    if ($hits) {
      $hits | ForEach-Object { Write-Error "$($_.Path):$($_.LineNumber):$($_.Line.Trim())" }
      throw "tracked leak pattern matched: $pattern"
    }
  }

  if (git ls-files .tmp/) {
    throw ".tmp/ must stay untracked"
  }

  cargo test --quiet
  if ($LASTEXITCODE -ne 0) { throw "cargo test failed" }

  Write-Host "publish preflight ok"
}
finally {
  Pop-Location
}
