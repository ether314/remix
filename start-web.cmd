@echo off
rem Start the Remix web app with one hour of Google access and open it in the browser.
rem Run it again whenever paid steps say Google access has run out.
cd /d "%~dp0"

set "GOOGLE_ACCESS_TOKEN="
for /f "delims=" %%t in ('gcloud.cmd auth print-access-token --quiet 2^>nul ^< nul') do set "GOOGLE_ACCESS_TOKEN=%%t"
for /f "delims=" %%p in ('gcloud.cmd config get-value project --quiet 2^>nul ^< nul') do set "GOOGLE_CLOUD_PROJECT=%%p"
if not defined GOOGLE_ACCESS_TOKEN echo Google sign-in has expired, so paid steps will be off. Run "gcloud.cmd auth login" and start again.

if /i not "%~1"=="--no-browser" start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8765"
docker compose up --force-recreate web
