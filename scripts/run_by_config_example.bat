:: Get YAML configs
::
:: Run from Ananconda prompt
::
:: Activate the fsctbase env:
::
::    conda activate fsctbase
::
::    cd \RemoteSensing\MLS\TreeTools\Scripts
::
::    python -m pip import pyyaml
::

set ARCPRO_PYTHON="C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"

set SURVEY123_GDB=V:\FCNSW_Hovermap_2023\Survey123\20231024.gdb

:: The survey 123 table name in the geodatabase
set FEATURE_CLASS=Hovermap_Survey_2023_FCNSW  

set LAZ_FOLDER=V:\FCNSW_Hovermap_2023\20231009

:: Creates YAML file
%ARCPRO_PYTHON% get_config_from_Survey123_FCNSW.py %SURVEY123_GDB% %FEATURE_CLASS% %LAZ_FOLDER%

:: Run TreeTools
python run_with_config.py -m -w -r MLS %LAZ_FOLDER%

