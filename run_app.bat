@echo off
rem — Перехід у теку з цим bat-файлом
cd /d "%~dp0"

rem — Активуємо базове Anaconda-середовище
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base

rem — Запускаємо Streamlit через python -m
python -m streamlit run streamlit_app.py

pause
