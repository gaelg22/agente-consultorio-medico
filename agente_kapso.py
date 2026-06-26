"""
agente_kapso.py — Agente IA del Consultorio Dr. Alejandro Martínez
Flujo: WhatsApp → Kapso Webhook → Agente RAG → Kapso API

Bases de datos:
  - MongoDB Atlas  : citas (guardar, consultar, disponibilidad)
  - PostgreSQL     : logs de conversaciones + feedback del doctor
  - vault/*.md     : personalidad y reglas del bot (lectura directa, sin PyTorch)
"""

import hashlib
import hmac
import json
import os
import re
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from rich.console import Console

from db_mongo    import init_mongo, guardar_cita, buscar_cita_existente, verificar_conflicto
from db_mongo    import horas_disponibles, proximas_disponibilidades
from db_postgres import init_postgres, log_mensaje
from admin       import router as admin_router

load_dotenv()
console = Console()

# ══════════════════════════════════════════════════════════════════
#  Configuración
# ══════════════════════════════════════════════════════════════════

NEGOCIO_NOMBRE      = "Consultorio Dr. Alejandro Martínez"
NEGOCIO_TIPO        = "Cirugía General"
AGENTE_NOMBRE       = "Mateo"
AGENTE_ROL          = "parte del equipo del consultorio"

# Carpeta con los .md del vault de Obsidian (incluida en el repo)
VAULT_PATH = Path(os.getenv("VAULT_DIR", "./vault"))

HORARIOS_DIA = {
    "lunes":     {"inicio": 9 * 60,  "fin": 14 * 60},
    "martes":    {"inicio": 16 * 60, "fin": 20 * 60},
    "miercoles": {"inicio": 9 * 60,  "fin": 14 * 60},
    "jueves":    {"inicio": 16 * 60, "fin": 20 * 60},
    "viernes":   {"inicio": 9 * 60,  "fin": 14 * 60},
    "sabado":    {"inicio": 9 * 60,  "fin": 13 * 60},
    "domingo":   None,
}

LLM_MODEL      = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024
MAX_HISTORIAL  = 20

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
KAPSO_API_KEY         = os.getenv("KAPSO_API_KEY", "")
KAPSO_PHONE_NUMBER_ID = os.getenv("KAPSO_PHONE_NUMBER_ID", "")
KAPSO_WEBHOOK_SECRET  = os.getenv("KAPSO_WEBHOOK_SECRET", "")
KAPSO_API_BASE        = "https://api.kapso.ai/v1"

GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN", "")
GITHUB_VAULT_OWNER = os.getenv("GITHUB_VAULT_OWNER", "gaelg22")
GITHUB_VAULT_REPO  = os.getenv("GITHUB_VAULT_REPO", "doctor-vault")

# ══════════════════════════════════════════════════════════════════
#  Constantes internas
# ══════════════════════════════════════════════════════════════════

DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

FRASES_CONFIRMACION = [
    "tu cita está registrada", "tu cita queda registrada",
    "quedas agendado", "quedas agendada", "cita confirmada",
    "ya quedaste agendado", "ya quedaste agendada",
    "te registramos", "te agendamos",
]

PALABRAS_AFIRMACION = [
    "si", "sí", "soy yo", "ese soy", "esa soy", "correcto",
    "exacto", "claro que si", "soy el mismo", "soy la misma", "asi es",
]
PALABRAS_NEGACION = [
    "no", "otra persona", "somos diferentes", "no soy yo",
    "diferente", "no es", "no, soy",
]

# Estado en memoria por usuario
historiales:      dict[str, list[dict]] = defaultdict(list)
citas_pendientes: dict[str, dict | None] = defaultdict(lambda: None)

# Contenido del vault cargado al inicio
vault_contenido: str = ""
cliente_ant: anthropic.Anthropic | None = None


# ══════════════════════════════════════════════════════════════════
#  Cargar vault de Obsidian
# ══════════════════════════════════════════════════════════════════

def cargar_vault() -> str:
    """Lee todos los .md del vault y los concatena como contexto del bot."""
    if not VAULT_PATH.exists():
        console.print(f"[yellow]Carpeta vault no encontrada en {VAULT_PATH}[/yellow]")
        return ""

    partes = []
    # Archivos principales
    for archivo in sorted(VAULT_PATH.glob("*.md")):
        contenido = archivo.read_text(encoding="utf-8")
        partes.append(f"## {archivo.stem}\n{contenido}")

    # Correcciones aprendidas
    correcciones_path = VAULT_PATH / "correcciones"
    if correcciones_path.exists():
        for archivo in sorted(correcciones_path.glob("*.md")):
            contenido = archivo.read_text(encoding="utf-8")
            partes.append(f"## corrección: {archivo.stem}\n{contenido}")

    resultado = "\n\n---\n\n".join(partes)
    console.print(f"[green]✅ Vault cargado: {len(partes)} archivos ({len(resultado)} chars)[/green]")
    return resultado


# ══════════════════════════════════════════════════════════════════
#  Lifespan
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vault_contenido, cliente_ant

    console.print("[cyan]Iniciando bases de datos...[/cyan]")
    await init_mongo()
    await init_postgres()

    console.print("[cyan]Cargando vault de Obsidian...[/cyan]")
    vault_contenido = cargar_vault()

    cliente_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    console.print(f"[bold green]🚀 {AGENTE_NOMBRE} ({NEGOCIO_NOMBRE}) listo[/bold green]")
    yield


app = FastAPI(title=f"Agente — {NEGOCIO_NOMBRE}", lifespan=lifespan)
app.include_router(admin_router)


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def detectar_confirmacion(texto: str) -> bool:
    t = texto.lower()
    return any(f in t for f in FRASES_CONFIRMACION)


def usuario_confirma(texto: str) -> bool | None:
    t = texto.lower()
    if any(p in t for p in PALABRAS_AFIRMACION):
        return True
    if any(p in t for p in PALABRAS_NEGACION):
        return False
    return None


async def extraer_datos_cita(conv: list[dict]) -> dict | None:
    texto_conv = "\n".join(
        f"{'Paciente' if m['role'] == 'user' else AGENTE_NOMBRE}: {m['content']}"
        for m in conv[-12:]
    )
    prompt = (
        "Extrae los datos de la cita médica de esta conversación.\n"
        "Devuelve SOLO un JSON con: nombre, fecha (YYYY-MM-DD), hora (HH:MM), motivo, tipo.\n"
        "Para 'tipo' usa solo 'primera vez' o 'seguimiento'.\n"
        "Si un dato no está claro, usa 'por definir'.\n\n"
        f"Conversación:\n{texto_conv}\n\nJSON:"
    )
    try:
        resp = cliente_ant.messages.create(
            model=LLM_MODEL, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw   = resp.content[0].text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def calcular_dias_proximos() -> str:
    hoy = date.today()
    fin = hoy + timedelta(days=90)
    dia = hoy
    rows = []
    while dia <= fin:
        nombre = DIAS_SEMANA[dia.weekday()]
        rango  = HORARIOS_DIA.get(nombre)
        if rango:
            h_ini = f"{rango['inicio'] // 60}:{rango['inicio'] % 60:02d}"
            h_fin = f"{rango['fin'] // 60}:{rango['fin'] % 60:02d}"
            rows.append(f"  {dia.isoformat()} ({nombre}): {h_ini}–{h_fin}")
        else:
            rows.append(f"  {dia.isoformat()} ({nombre}): CERRADO")
        dia += timedelta(days=1)
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════
#  Kapso API
# ══════════════════════════════════════════════════════════════════

async def enviar_whatsapp(to_phone: str, texto: str) -> bool:
    url = f"{KAPSO_API_BASE}/whatsapp/phone-numbers/{KAPSO_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {KAPSO_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {"to": to_phone, "type": "text", "text": {"body": texto}}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=headers, json=payload)
            return resp.status_code in (200, 201)
    except Exception as e:
        console.print(f"[red]Error enviando WhatsApp: {e}[/red]")
        return False


# ══════════════════════════════════════════════════════════════════
#  Lógica principal
# ══════════════════════════════════════════════════════════════════

async def procesar_mensaje(phone: str, mensaje: str) -> str:
    historial      = historiales[phone]
    cita_pendiente = citas_pendientes[phone]

    inicio           = time.time()
    omitir_deteccion = False
    nota_pendiente   = ""

    # Log del mensaje a PostgreSQL
    await log_mensaje(phone, "user", mensaje)

    # --- 1. Resolver confirmación de cita duplicada ---
    if cita_pendiente:
        respuesta_usuario = usuario_confirma(mensaje)
        if respuesta_usuario is True:
            try:
                doc = await guardar_cita(cita_pendiente, telefono=phone)
                nota_pendiente = (
                    f"ACCIÓN COMPLETADA: El paciente confirmó SÍ ser la misma persona. "
                    f"Cita de {doc.get('nombre','?')} el {doc.get('fecha','?')} "
                    f"a las {doc.get('hora','?')} guardada. Confirma con 'tu cita está registrada'.\n\n"
                )
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            citas_pendientes[phone] = None
            omitir_deteccion = True

        elif respuesta_usuario is False:
            try:
                doc = await guardar_cita(cita_pendiente, telefono=phone)
                nota_pendiente = (
                    f"ACCIÓN COMPLETADA: Es persona diferente. "
                    f"Cita de {doc.get('nombre','?')} guardada. Atiende al nuevo paciente.\n\n"
                )
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            citas_pendientes[phone] = None
            omitir_deteccion = True

    # --- 2. Detectar confirmación y guardar en Mongo ---
    if not omitir_deteccion and detectar_confirmacion(
        historial[-1]["content"] if historial else ""
    ):
        historial.append({"role": "user", "content": mensaje})
        datos = await extraer_datos_cita(historial)
        if datos and datos.get("fecha", "por definir") != "por definir":
            conflicto = await verificar_conflicto(datos.get("fecha", ""), datos.get("hora", ""))
            if conflicto:
                citas_pendientes[phone] = datos
                nota_pendiente = (
                    f"ALERTA: Ya hay cita el {conflicto['fecha']} a las {conflicto['hora']}. "
                    f"Pregunta amablemente si es la misma persona.\n\n"
                )
            else:
                try:
                    doc = await guardar_cita(datos, telefono=phone)
                    nota_pendiente = (
                        f"CITA GUARDADA: {doc.get('nombre','?')} "
                        f"el {doc.get('fecha','?')} a las {doc.get('hora','?')}. "
                        f"Confirma con 'tu cita está registrada'.\n\n"
                    )
                except Exception as e:
                    nota_pendiente = f"Error guardando cita: {e}\n\n"
    else:
        historial.append({"role": "user", "content": mensaje})

    if len(historial) > MAX_HISTORIAL:
        historiales[phone] = historial[-MAX_HISTORIAL:]
        historial = historiales[phone]

    # --- 3. Disponibilidad en tiempo real desde MongoDB ---
    disponibilidad = await proximas_disponibilidades(5)

    # --- 4. Construir system prompt con vault completo ---
    hoy          = date.today()
    dia_hoy      = DIAS_SEMANA[hoy.weekday()]
    fecha_maxima = hoy + timedelta(days=90)

    system_prompt = (
        f"Eres {AGENTE_NOMBRE}, {AGENTE_ROL} del {NEGOCIO_NOMBRE} ({NEGOCIO_TIPO}).\n"
        f"Hablas en español, con tono tranquilo, cálido y empático.\n"
        f"Nunca dices que eres una IA. Eres {AGENTE_NOMBRE}, del equipo.\n\n"

        f"⚠️ NUNCA TE PRESENTES NI DIGAS TU NOMBRE al inicio de cada mensaje.\n"
        f"Si el paciente dice 'Hola', responde con calidez pero SIN decir tu nombre.\n\n"

        "⚠️ FRASES PROHIBIDAS: 'Te entiendo', 'Entiendo perfectamente', 'Entiendo tu preocupación'.\n"
        "Usa: 'Qué bueno que nos contactas', 'Es válido que te preocupe', 'Eso suena incómodo'.\n\n"

        "⚡ MODO AGENDAMIENTO DIRECTO:\n"
        "Si el paciente pide cita → muestra horarios de inmediato.\n"
        "NO preguntes síntomas cuando ya decidió venir.\n\n"

        "FLUJO DE SÍNTOMAS (solo si reporta molestia sin pedir cita):\n"
        "En UN mensaje: valida + pregunta si tiene tratamiento.\n"
        "Si no mejora → recomienda cita Y muestra horarios en ESE MISMO mensaje.\n\n"

        "TONO: Prosa natural. La frase 'tu cita está registrada' debe aparecer siempre al confirmar.\n\n"

        "REGLAS MÉDICAS:\n"
        "1. NUNCA proporciones diagnósticos ni consejos de tratamiento.\n"
        "2. El doctor atiende CUALQUIER motivo. NUNCA redirigir a otro especialista.\n"
        "3. URGENCIAS: 'Acude a urgencias o llama al 911.'\n\n"

        f"FECHA DE HOY: {hoy.isoformat()} ({dia_hoy}).\n"
        f"DISPONIBILIDAD PRÓXIMA:\n{disponibilidad}\n\n"
        f"CALENDARIO 90 DÍAS:\n{calcular_dias_proximos()}\n\n"
        f"FECHA MÁXIMA: {fecha_maxima.isoformat()}.\n\n"

        f"{nota_pendiente}"
        "═══ INFORMACIÓN DEL CONSULTORIO (usa SOLO esto para responder) ═══\n\n"
        f"{vault_contenido}"
    )

    # --- 5. LLM ---
    try:
        resp = cliente_ant.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=system_prompt,
            messages=historial,
        )
        texto_respuesta = resp.content[0].text
    except Exception as e:
        texto_respuesta = "Disculpa, hubo un problema técnico. Por favor intenta de nuevo."
        console.print(f"[red]Error LLM: {e}[/red]")

    historial.append({"role": "assistant", "content": texto_respuesta})
    await log_mensaje(phone, "assistant", texto_respuesta)

    ms = int((time.time() - inicio) * 1000)
    console.print(f"[dim]{phone} → {ms}ms[/dim]")
    return texto_respuesta


# ══════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════

def verificar_firma_kapso(body: bytes, signature: str) -> bool:
    # Verificación desactivada temporalmente - Kapso usa formato de firma propio
    return True


@app.get("/webhook")
async def webhook_verify(request: Request):
    params    = dict(request.query_params)
    challenge = params.get("hub.challenge") or params.get("challenge")
    if challenge:
        return JSONResponse(content={"challenge": challenge})
    return JSONResponse(content={"status": "ok"})


@app.post("/webhook")
async def webhook_recibir(request: Request):
    body_bytes = await request.body()

    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido")

    # Log completo para diagnóstico
    console.print(f"[yellow]📩 Kapso payload recibido[/yellow]")
    console.print(f"[dim]{json.dumps(payload)[:600]}[/dim]")

    # ── Formato Kapso v2 ─────────────────────────────────────────
    # {
    #   "message": {
    #     "from": "5249...",
    #     "kapso": { "type": "text", "text": {"body": "..."}, "content": "..." }
    #   }
    # }
    msg = payload.get("message", {})
    if not msg:
        return JSONResponse(content={"status": "no message field"})

    phone = str(msg.get("from", "")).strip()

    # type y text están en message, NO dentro de kapso
    msg_type  = msg.get("type", "") or msg.get("kapso", {}).get("type", "")
    kapso     = msg.get("kapso", {})
    texto_obj = msg.get("text") or kapso.get("text") or {}

    if msg_type != "text":
        console.print(f"[yellow]Tipo ignorado: {msg_type!r}[/yellow]")
        return JSONResponse(content={"status": f"tipo ignorado: {msg_type}"})

    mensaje = (
        texto_obj.get("body")
        or kapso.get("content")
        or ""
    ).strip()

    if not phone or not mensaje:
        console.print(f"[yellow]Mensaje vacío: phone={phone!r} msg={mensaje!r}[/yellow]")
        return JSONResponse(content={"status": "empty"})

    console.print(f"[cyan]📱 {phone}:[/cyan] {mensaje[:80]}")
    respuesta = await procesar_mensaje(phone, mensaje)
    await enviar_whatsapp(phone, respuesta)
    return JSONResponse(content={"status": "ok"})


@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "negocio":          NEGOCIO_NOMBRE,
        "agente":           AGENTE_NOMBRE,
        "usuarios_activos": len(historiales),
        "vault_chars":      len(vault_contenido),
        "dbs":              ["mongodb", "postgresql"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    console.print(f"[bold green]🚀 {AGENTE_NOMBRE} — {NEGOCIO_NOMBRE}[/bold green]")
    uvicorn.run("agente_kapso:app", host="0.0.0.0", port=port, reload=False)
