copy "%CONDA_LIB_PATH%\libiomp*5md.dll" %PYTORCH_ROOT%\torch\lib
:: Should be set in build_pytorch.bat
echo "CURRENT DIRECTORY: "
echo %cd%
copy "%libuv_ROOT%\bin\uv.dll" %PYTORCH_ROOT%\torch\lib
