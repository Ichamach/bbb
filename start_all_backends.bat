@echo off
echo ============================================================
echo   NeuroViral Lab - Starting All Backends
echo ============================================================
echo.

REM Install dependencies first
echo Installing dependencies...
pip install fastapi uvicorn numpy scipy pydantic
echo.

REM Start each backend in a new window
echo Starting BBB engine on port 8000...
start "BBB Engine - Port 8000" cmd /k "cd /d %~dp0 && python -m uvicorn main:app --port 8000"

timeout /t 2 /nobreak > nul

echo Starting RABV engine on port 8001...
start "RABV Engine - Port 8001" cmd /k "cd /d %~dp0 && python -m uvicorn rabv_main:app --port 8001"

timeout /t 2 /nobreak > nul

echo Starting Split-Protein Designer on port 8002...
start "Split Designer - Port 8002" cmd /k "cd /d %~dp0 && python -m uvicorn split_main:app --port 8002"

timeout /t 2 /nobreak > nul

echo Starting Protein Splitter on port 8003...
start "Protein Splitter - Port 8003" cmd /k "cd /d %~dp0 && python -m uvicorn splitter_main:app --port 8003"

timeout /t 2 /nobreak > nul

echo Starting Static File Server on port 3000...
start "Static Server - Port 3000" cmd /k "cd /d %~dp0 && python -m http.server 3000"

timeout /t 3 /nobreak > nul

echo.
echo ============================================================
echo   All backends started! Open your browser and go to:
echo.
echo   http://localhost:3000/bbb_simulator.html
echo   http://localhost:3000/rabies_simulator.html
echo   http://localhost:3000/split_protein_simulator.html
echo   http://localhost:3000/protein_splitter.html
echo ============================================================
echo.
echo Press any key to open the BBB simulator in your browser...
pause > nul
start http://localhost:3000/bbb_simulator.html
