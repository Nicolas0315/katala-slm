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

  # Match the shape of a home path, never a spelled-out one. Naming the operator's
  # real directories here published exactly what the check exists to keep out, and
  # the script excluded itself from the scan, so it was the one file that could
  # leak without the check ever seeing it.
  $patterns = @(
    '[A-Za-z]:[\\/][Uu]sers[\\/][A-Za-z][A-Za-z0-9_-]{2,}',
    '/(Users|home)/[A-Za-z][A-Za-z0-9_-]{2,}',
    '100\.\d+\.\d+\.\d+',
    'sk-[A-Za-z0-9_-]{20,}',
    'ghp_[A-Za-z0-9_]{20,}',
    'BEGIN (RSA |OPENSSH )?PRIVATE KEY'
  )

  # Documented placeholders are how a doc is supposed to write these.
  $allowed = '[\\/][Uu]sers[\\/](youruser|me|\$)|/(Users|home)/(youruser|me|runner|\$)|\$env:USERPROFILE|\$HOME'

  # No self-exclusion: the patterns above are character classes, so this file does
  # not match itself, and scanning it is what would have caught the old literals.
  $files = git ls-files
  foreach ($pattern in $patterns) {
    $hits = $files | ForEach-Object {
      if (Test-Path -LiteralPath $_) {
        Select-String -Path $_ -Pattern $pattern -ErrorAction SilentlyContinue
      }
    }
    $hits = $hits | Where-Object { $_.Line -notmatch $allowed }
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
