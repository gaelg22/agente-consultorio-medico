"""
agente_kapso.py — Agente IA del Consultorio Dr. Alejandro Martínez
Flujo: WhatsApp → Kapso Webhook → Agente RAG → Kapso API

Bases de datos:
  - MongoDB Atlas  : citas (guardar, consultar, disponibilidad)
  - PostgreSQL     : logs de conversaciones + feedback del doctor
  - Obsidian/ChromaDB: personalidad y reglas del bot (RAG)
"""

import base64
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
import chromadb
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from rich.console import Console
from sentence_transformers import SentenceTransformer

from db_mongo    import init_mongo, guardar_cita, buscar_cita_existente, verificar_conflicto
from db_mongo    import horas_disponibles, proximas_disponibilidades
from db_postgres import init_postgres, log_mensaje
from admin       import router as admin_router

load_dotenv()
console = Console()

# ══════════════════════════════════════════════════════════════════
#  Configuración del negocio
# ══════════════════════════════════════════════════════════════════

NEGOCIO_NOMBRE      = "Consultorio Dr. Alejandro Martínez"
NEGOCIO_TIPO        = "Cirugía General"
AGENTE_NOMBRE       = "Mateo"
AGENTE_ROL          = "parte del equipo del consultorio"

CHROMA_PATH = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

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

# Historial por teléfono
historiales:      dict[str, list[dict]] = defaultdict(list)
citas_pendientes: dict[str, dict | None] = defaultdict(lambda: None)

# Clientes globales
modelo_embeddings: SentenceTransformer | None = None
collection = None
cliente_ant: anthropic.Anthropic | None = None


# ══════════════════════════════════════════════════════════════════
#  Lifespan — arranque del servidor
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo_embeddings, collection, cliente_ant

    console.print("[cyan]Iniciando bases de datos...[/cyan]")
    await init_mongo()
    await init_postgres()

    console.print("[cyan]Cargando modelo de embeddings (Obsidian RAG)...[/cyan]")
    modelo_embeddings = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection    = chroma_client.get_or_create_collection("documentos")

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


def buscar_contexto(query: str) -> tuple[str, list[str]]:
    """RAG: busca en ChromaDB los fragmentos de Obsidian más relevantes."""
    if modelo_embeddings is None or collection is None:
        return "", []
    emb     = modelo_embeddings.encode([query]).tolist()
    results = collection.query(query_embeddings=emb, n_results=5)
    docs    = results.get("documents", [[]])[0]
    ids     = results.get("ids", [[]])[0]
    partes, fuentes = [], []
    for doc, doc_id in zip(docs, ids):
        if doc and doc.strip():
            partes.append(f"## {doc_id}\n{doc}")
            fuentes.append(doc_id)
    return "\n\n".join(partes), fuentes


async def extraer_datos_cita(conv: list[dict]) -> dict | None:
    texto_conv = "\n".join(
        f"{'Paciente' if m['role'] == 'user' else AGENTE_NOMBRE}: {m['content']}"
        for m in conv[-12:]
    )
    prompt = (
        f"Extrae los datos de la cita médica de esta conversación.\n"
        f"Devuelve SOLO un JSON con: nombre, fecha (YYYY-MM-DD), hora (HH:MM), motivo, tipo.\n"
        f"Para 'tipo' usa solo 'primera vez' o 'seguimiento'.\n"
        f"Si un dato no está claro, usa 'por definir'.\n\n"
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
    hoy  = date.today()
    fin  = hoy + timedelta(days=90)
    dia  = hoy
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
#  Envío de mensajes via Kapso API
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
#  Lógica principal del agente
# ══════════════════════════════════════════════════════════════════

async def procesar_mensaje(phone: str, mensaje: str) -> str:
    historial      = historiales[phone]
    cita_pendiente = citas_pendientes[phone]

    inicio           = time.time()
    omitir_deteccion = False
    nota_pendiente   = ""

    # Log del mensaje del usuario a PostgreSQL
    await log_mensaje(phone, "user", mensaje)

    # --- 1. Resolver confirmación de cita duplicada ---
    if cita_pendiente:
        respuesta_usuario = usuario_confirma(mensaje)
        if respuesta_usuario is True:
            try:
                doc = await guardar_cita(cita_pendiente, telefono=phone)
                nota_pendiente = (
                    f"ACCIÓN COMPLETADA: El paciente confirmó que SÍ es la misma persona. "
                    f"La cita de {doc.get('nombre','?')} para el {doc.get('fecha','?')} "
                    f"a las {doc.get('hora','?')} ya fue guardada. "
                    f"Confirma con 'tu cita está registrada'.\n\n"
                )
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            citas_pendientes[phone] = None
            omitir_deteccion = True

        elif respuesta_usuario is False:
            try:
                doc = await guardar_cita(cita_pendiente, telefono=phone)
                nota_pendiente = (
                    f"ACCIÓN COMPLETADA: Es una persona diferente. "
                    f"Cita de {doc.get('nombre','?')} guardada. Ahora atiende al nuevo paciente.\n\n"
                )
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            citas_pendientes[phone] = None
            omitir_deteccion = True

    # --- 2. RAG: buscar contexto en ChromaDB (Obsidian) ---
    contexto, fuentes = buscar_contexto(mensaje)

    # --- 3. Detectar confirmación y guardar cita en Mongo ---
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
                    f"ALERTA DE DUPLICADO: Ya hay una cita el {conflicto['fecha']} "
                    f"a las {conflicto['hora']}. Pregunta amablemente si es la misma persona.\n\n"
                )
            else:
                try:
                    doc = await guardar_cita(datos, telefono=phone)
                    nota_pendiente = (
                        f"CITA GUARDADA en MongoDB: {doc.get('nombre','?')} "
                        f"el {doc.get('fecha','?')} a las {doc.get('hora','?')}. "
                        f"Confirma con 'tu cita está registrada'.\n\n"
                    )
                except Exception as e:
                    nota_pendiente = f"Error guardando cita: {e}\n\n"
    else:
        historial.append({"role": "user", "content": mensaje})

    # Truncar historial
    if len(historial) > MAX_HISTORIAL:
        historiales[phone] = historial[-MAX_HISTORIAL:]
        historial = historiales[phone]

    # --- 4. Disponibilidad en tiempo real desde MongoDB ---
    disponibilidad = await proximas_disponibilidades(5)

    # --- 5. Construir system prompt ---
    hoy          = date.today()
    dia_hoy      = DIAS_SEMANA[hoy.weekday()]
    fecha_maxima = hoy + timedelta(days=90)

    system_prompt = (
        f"Eres {AGENTE_NOMBRE}, {AGENTE_ROL} del {NEGOCIO_NOMBRE} ({NEGOCIO_TIPO}).\n"
        f"Hablas en español, con tono tranquilo, cálido y empático. Como alguien de confianza.\n"
        f"Nunca dices que eres una IA. Eres {AGENTE_NOMBRE}, del equipo.\n\n"

        f"⚠️ REGLA ABSOLUTA — NUNCA TE PRESENTES NI DIGAS TU NOMBRE:\n"
        f"El paciente ya sabe quién eres. PROHIBIDO repetir '{AGENTE_NOMBRE}', 'Soy {AGENTE_NOMBRE}', etc.\n"
        f"Si el paciente dice 'Hola', responde con calidez pero SIN decir tu nombre.\n\n"

        "⚠️ FRASES PROHIBIDAS:\n"
        "'Te entiendo', 'Entiendo perfectamente', 'Entiendo tu preocupación'.\n"
        "Usa en su lugar: 'Qué bueno que nos contactas', 'Es válido que te preocupe',\n"
        "  'Eso que describes suena incómodo', 'Qué difícil que siga el dolor'.\n\n"

        "═══ FLUJO DE ATENCIÓN ═══\n"
        "Sé breve y conversacional. Habla como persona, no como formulario.\n\n"

        "⚡ MODO AGENDAMIENTO DIRECTO:\n"
        "Si el paciente pide cita directamente → muestra horarios de inmediato.\n"
        "NO preguntes síntomas ni tratamiento cuando ya decidió venir.\n"
        "Si menciona motivo durante agendamiento → acéptalo y confirma. No preguntes '¿ya tienes tratamiento?'.\n\n"

        "FLUJO DE SÍNTOMAS (solo si reporta molestia sin pedir cita):\n"
        "PASO 1+2: En UN mensaje: valida brevemente + pregunta si tiene tratamiento.\n"
        "PASO 3: Si tiene tratamiento y está estable → tranquiliza y cierra.\n"
        "Si no mejora → recomienda cita Y muestra horarios en ESE MISMO mensaje.\n\n"

        "TONO: Prosa natural, sin listas de campos (Nombre:, Fecha:) en confirmaciones.\n"
        "La confirmación debe sonar como persona real. La frase 'tu cita está registrada' debe aparecer siempre.\n\n"

        "RECONOCIMIENTO DEL NOMBRE:\n"
        "• Ignora artículos (el, la). 'Soy el Luis' → 'Luis'.\n"
        "• 2+ palabras = nombre completo. Solo pide apellido si dio UNA sola palabra.\n"
        "• Si el paciente ya dio nombre en la conversación, NO lo pidas de nuevo.\n\n"

        "TIPO DE CITA:\n"
        "Si mencionó visita previa o receta → SEGUIMIENTO. Si no hay evidencia → primera vez.\n\n"

        "MOTIVO: Usa lo que el paciente describió. NUNCA inventes. Si no mencionó nada → 'por definir'.\n\n"

        "REGLAS MÉDICAS:\n"
        "1. NUNCA proporciones diagnósticos ni consejos de tratamiento.\n"
        "2. El doctor atiende CUALQUIER motivo, no solo cirugía. NUNCA redirigir a otro especialista.\n"
        "3. URGENCIAS: 'Acude a urgencias o llama al 911.'\n\n"

        "REGLA DE DÍA CERRADO: Si piden domingo → ofrece sábado/lunes con sus horas disponibles.\n\n"

        f"FECHA DE HOY: {hoy.isoformat()} ({dia_hoy}).\n"
        f"DISPONIBILIDAD PRÓXIMA (en tiempo real desde base de datos):\n{disponibilidad}\n\n"
        f"CALENDARIO 90 DÍAS:\n{calcular_dias_proximos()}\n\n"
        f"FECHA MÁXIMA: {fecha_maxima.isoformat()}.\n\n"

        f"{nota_pendiente}"
        "Usa solo la siguiente información del consultorio para responder:\n\n"
        f"{contexto}"
    )

    # --- 6. Llamar al LLM ---
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

    # Log de la respuesta del bot a PostgreSQL
    await log_mensaje(phone, "assistant", texto_respuesta)

    ms = int((time.time() - inicio) * 1000)
    console.print(f"[dim]{phone} → {ms}ms | fuentes Obsidian: {fuentes}[/dim]")
    return texto_respuesta


# ══════════════════════════════════════════════════════════════════
#  Endpoints FastAPI
# ══════════════════════════════════════════════════════════════════

def verificar_firma_kapso(body: bytes, signature: str) -> bool:
    if not KAPSO_WEBHOOK_SECRET:
        return True
    expected = hmac.new(
        KAPSO_WEBHOOK_SECRET.encode(), body, hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


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
    sig        = request.headers.get("X-Kapso-Signature", "")
    if not verificar_firma_kapso(body_bytes, sig):
        raise HTTPException(status_code=401, detail="Firma inválida")

    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido")

    if payload.get("event") != "whatsapp.message.received":
        return JSONResponse(content={"status": "ignored"})

    data  = payload.get("data", {})
    if data.get("type") != "text":
        return JSONResponse(content={"status": "non-text ignored"})

    phone   = data.get("from", "")
    mensaje = data.get("text", {}).get("body", "").strip()
    if not phone or not mensaje:
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
        "dbs":              ["mongodb", "postgresql", "chromadb"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    console.print(f"[bold green]🚀 {AGENTE_NOMBRE} — {NEGOCIO_NOMBRE}[/bold green]")
    uvicorn.run("agente_kapso:app", host="0.0.0.0", port=port, reload=False)
