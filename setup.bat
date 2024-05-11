@echo off

rem Define directories
set "RawDir=dev_data\raw"
set "ProcessedDir=dev_data\processed"

rem Create directories if they don't exist
mkdir "%RawDir%" 2>nul
mkdir "%ProcessedDir%" 2>nul

rem Download dataset
curl -L "https://zenodo.org/records/7882613/files/dev_gearbox.zip?download=1" -o "%RawDir%\gearbox.zip"

rem Unzip dataset
powershell Expand-Archive -Path "%RawDir%\gearbox.zip" -DestinationPath "%RawDir%"

rem Remove the downloaded zip file
del "%RawDir%\gearbox.zip"

rem Move train data to appropriate directory
move "%RawDir%\gearbox\train" "%RawDir%\gearbox\normal"
