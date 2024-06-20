@echo off

REM Define the range and step
set start=425
set end=500
set step=25

REM Define the log files
set log_file=script.log
set python_log_file=python.log

REM Clear the log files at the start
echo. > %log_file%
echo. > %python_log_file%

REM Loop through the iterations
for /l %%i in (%start%, %step%, %end%) do (
    echo Running iteration %%i >> %log_file% 2>&1
    REM Run the Python script and log the output
    python main.py --mask_factor %%i >> %python_log_file% 2>&1

    REM Check if the Python script ran successfully
    if errorlevel 1 (
        echo Error occurred in iteration %%i, exiting... >> %log_file% 2>&1
        exit /b 1
    )

    echo Iteration %%i completed >> %log_file% 2>&1

    REM Add all changes to git and log the output
    git add . >> %log_file% 2>&1
    git commit -m "Iteration %%i completed" >> %log_file% 2>&1
    git push origin main >> %log_file% 2>&1
    echo Iteration %%i results pushed to GitHub >> %log_file% 2>&1
)

echo All iterations completed. >> %log_file% 2>&1
