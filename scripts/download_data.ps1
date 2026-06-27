# Windows PowerShell equivalent of download_data.sh
# Prerequisites: pip install kaggle, set KAGGLE_USERNAME and KAGGLE_KEY
#   or place kaggle.json at $env:USERPROFILE\.kaggle\kaggle.json
# Usage: .\scripts\download_data.ps1

$competition = "aptos2019-blindness-detection"
$dataDir     = "data"

Write-Host "Downloading competition: $competition"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

kaggle competitions download -c $competition -p $dataDir

Write-Host "Unzipping..."
Expand-Archive -Path "$dataDir\$competition.zip" -DestinationPath $dataDir -Force
Remove-Item "$dataDir\$competition.zip"

Write-Host "Done. Contents of $dataDir\:"
Get-ChildItem $dataDir | Select-Object Name
