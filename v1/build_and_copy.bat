@echo off
cd /d "%~dp0"
echo === Building SNA C++ extension ===
cmake --build build\temp --config Release
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED!
    pause
    exit /b 1
)
echo === Copying to install locations ===
copy /Y "python\Release\core_cpp.cp311-win_amd64.pyd" "python\core_cpp.cp311-win_amd64.pyd"
copy /Y "python\Release\core_cpp.cp311-win_amd64.pyd" "core_cpp.cp311-win_amd64.pyd"
echo === Build complete ===