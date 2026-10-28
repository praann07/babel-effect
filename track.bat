@echo off
cd /d "%~dp0"
set PYTHONPATH=code
set PYTHONIOENCODING=utf8
title Night run tracker
.venv\Scripts\python.exe code\track.py
