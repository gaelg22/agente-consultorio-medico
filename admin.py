"""
admin.py — Panel de administración para el Dr. Alejandro Martínez.
Permite revisar conversaciones, marcar errores del bot y generar
correcciones automáticas en Obsidian (doctor-vault en GitHub).
"""

import base64
import hashlib
import os
import secrets
from datetime import date
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from rich.console import Console

from db_postgres import get_mensajes_sesion, get_sesiones_recientes, guardar_feedback

console = Console()
router  = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBasic()

ADMIN_PASSWORD     = os.getenv("ADMIN_PASSWORD", "")
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN", "")
GITHUB_VAULT_OWNER = os.getenv("GITHUB_VAULT_OWNER", "gaelg22")
GITHUB_VAULT_REPO  = os.getenv("GITHUB_VAULT_REPO", "doctor-vault")


# ── Auth básica ────────────────────────────────────────────────────

def verificar_admin(credentials: HTTPBasicCredentials = Depends(security)):
    # sin ADMIN_PASSWORD configurado el panel queda cerrado
    pwd_ok = bool(ADMIN_PASSWORD) and secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not pwd_ok:
        raise HTTPException(
            status_code=401,
            detail="Credenciales inválidas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ── GitHub: crear nota de corrección ──────────────────────────────

async def crear_nota_correccion(feedback_id: int, respuesta_mala: str, correccion: str) -> str:
    """Genera un .md de corrección en doctor-vault/correcciones/ vía GitHub API."""
    if not GITHUB_TOKEN:
        return ""

    hoy           = date.today().isoformat()
    nombre_archivo = f"correccion-{hoy}-{feedback_id:04d}.md"
    ruta_github    = f"correcciones/{nombre_archivo}"

    contenido = (
        f"---\n"
        f"tipo: correccion\n"
        f"fecha: {hoy}\n"
        f"---\n\n"
        f"# Corrección #{feedback_id}\n\n"
        f"## Respuesta incorrecta del bot\n\n"
        f"{respuesta_mala}\n\n"
        f"## Cómo debió responder\n\n"
        f"{correccion}\n\n"
        f"## Nota para el bot\n\n"
        f"Cuando el paciente pregunte algo similar a la situación descrita arriba, "
        f"responde siguiendo el ejemplo de la corrección.\n"
    )

    url = f"https://api.github.com/repos/{GITHUB_VAULT_OWNER}/{GITHUB_VAULT_REPO}/contents/{ruta_github}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "message": f"Corrección bot #{feedback_id} — {hoy}",
        "content": base64.b64encode(contenido.encode("utf-8")).decode("ascii"),
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(url, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                console.print(f"[green]✅ Nota de corrección creada: {nombre_archivo}[/green]")
                return nombre_archivo
            else:
                console.print(f"[red]Error GitHub: {resp.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]Error creando nota: {e}[/red]")
    return ""


# ── Endpoints ──────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def panel_admin(_: str = Depends(verificar_admin)):
    html = Path("admin.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@router.get("/sesiones")
async def sesiones(_: str = Depends(verificar_admin)):
    return await get_sesiones_recientes(30)


@router.get("/sesion/{session_id}")
async def mensajes_sesion(session_id: str, _: str = Depends(verificar_admin)):
    return await get_mensajes_sesion(session_id)


class FeedbackRequest(BaseModel):
    session_id:    str
    respuesta_mala: str
    contexto:      str = ""
    correccion:    str


@router.post("/feedback")
async def enviar_feedback(req: FeedbackRequest, _: str = Depends(verificar_admin)):
    """El doctor marca una respuesta mala y escribe la corrección."""
    # 1. Guardar en PostgreSQL
    fb_id = await guardar_feedback(
        session_id=req.session_id,
        respuesta_mala=req.respuesta_mala,
        correccion=req.correccion,
        contexto=req.contexto,
    )

    # 2. Crear nota en Obsidian (GitHub)
    nota = await crear_nota_correccion(fb_id, req.respuesta_mala, req.correccion)

    return {
        "ok": True,
        "feedback_id": fb_id,
        "nota_obsidian": nota,
        "mensaje": "✅ Corrección guardada. El bot aprenderá en el próximo re-índice.",
    }
