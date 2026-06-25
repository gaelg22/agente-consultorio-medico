# sync-obsidian.ps1
# Ejecuta este script para traer las nuevas citas de WhatsApp a tu Obsidian.
# Puedes programarlo en el Programador de tareas de Windows para que corra cada 5 minutos.

$vaultPath = "C:\Users\gaelg_12gxbrs\OneDrive\Documentos\Doctor-Agente\Doctor-Agente"

Write-Host "Sincronizando vault de Obsidian con GitHub..." -ForegroundColor Cyan
Set-Location $vaultPath
git pull origin main
Write-Host "Listo. Nuevas citas disponibles en Obsidian." -ForegroundColor Green
