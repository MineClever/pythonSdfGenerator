chcp 936
@echo off
cls

:start
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
goto :eof

pause