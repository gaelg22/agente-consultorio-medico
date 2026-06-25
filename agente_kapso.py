"""
╔══════════════════════════════════════════════════════════════════╗
║         AGENTE RAG — TEMPLATE KAPSO (WhatsApp Business)         ║
║  Adapta este archivo para cualquier negocio que use Kapso.       ║
║  Flujo: WhatsApp → Kapso Webhook → Agente RAG → Kapso API        ║
╚══════════════════════════════════════════════════════════════════╝

CONFIGURACIÓN RÁPIDA:
  1. Copia y ajusta las variables de la sección ═══ CONFIGURACIÓN ═══
  2. Pon tus keys en el archivo .env
  3. Apunta el webhook de Kapso a: https://tu-servidor.com/webhook
  4. Ejecuta: python agente_kapso.py
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

load_dotenv()
console = Console()

# ══════════════════════════════════════════════════════════════════
#  ═══ CONFIGURACIÓN — Cambia esto por tu negocio ═══
# ══════════════════════════════════════════════════════════════════

NEGOCIO_NOMBRE      = "Consultorio Dr. Alejandro Martínez"
NEGOCIO_TIPO        = "Cirugía General"
AGENTE_NOMBRE       = "Mateo"          # Nombre del agente de atención
AGENTE_ROL          = "parte del equipo del consultorio"

# Rutas — leen de variables de entorno para compatibilidad con Render/cloud
VAULT_PATH  = Path(os.getenv("CITAS_DIR", "./citas"))
CITAS_PATH  = VAULT_PATH
CHROMA_PATH = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

# Horarios por día de semana (None = cerrado)
HORARIOS_DIA = {
    "lunes":     {"inicio": 9 * 60,  "fin": 14 * 60},
    "martes":    {"inicio": 16 * 60, "fin": 20 * 60},
    "miercoles": {"inicio": 9 * 60,  "fin": 14 * 60},
    "jueves":    {"inicio": 16 * 60, "fin": 20 * 60},
    "viernes":   {"inicio": 9 * 60,  "fin": 14 * 60},
    "sabado":    {"inicio": 9 * 60,  "fin": 13 * 60},
    "domingo":   None,
}

# Modelo de lenguaje (Anthropic)
LLM_MODEL    = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024

# Historial máximo por usuario (turnos)
MAX_HISTORIAL = 20

# ══════════════════════════════════════════════════════════════════
#  Variables de entorno requeridas en .env:
#    ANTHROPIC_API_KEY=sk-ant-...
#    KAPSO_API_KEY=kso_...
#    KAPSO_PHONE_NUMBER_ID=...        (ID del número en Kapso)
#    KAPSO_WEBHOOK_SECRET=...         (para verificar firmas, opcional)
# ══════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
KAPSO_API_KEY          = os.getenv("KAPSO_API_KEY", "")
KAPSO_PHONE_NUMBER_ID  = os.getenv("KAPSO_PHONE_NUMBER_ID", "")
KAPSO_WEBHOOK_SECRET   = os.getenv("KAPSO_WEBHOOK_SECRET", "")

KAPSO_API_BASE = "https://api.kapso.ai/v1"

# ── GitHub Vault Sync ──────────────────────────────────────────────
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
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

# Historial por número de teléfono  {phone: [{"role":..., "content":...}]}
historiales: dict[str, list[dict]] = defaultdict(list)
# Cita pendiente de confirmación por usuario  {phone: dict|None}
citas_pendientes: dict[str, dict | None] = defaultdict(lambda: None)


# ══════════════════════════════════════════════════════════════════
#  Inicialización del modelo y ChromaDB
# ══════════════════════════════════════════════════════════════════

modelo_embeddings: SentenceTransformer | None = None
collection = None
cliente_ant: anthropic.Anthropic | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo_embeddings, collection, cliente_ant
    console.print(f"[bold cyan]Cargando modelo de embeddings...[/bold cyan]")
    modelo_embeddings = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    cliente_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = cliente_chroma.get_or_create_collection("documentos")
    cliente_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    console.print(f"[bold green]{AGENTE_NOMBRE} ({NEGOCIO_NOMBRE}) listo para WhatsApp via Kapso[/bold green]")
    yield


app = FastAPI(title=f"Agente Kapso — {NEGOCIO_NOMBRE}", lifespan=lifespan)


# ══════════════════════════════════════════════════════════════════
#  Helpers de citas
# ══════════════════════════════════════════════════════════════════

def parsear_frontmatter(texto: str) -> dict:
    match = re.search(r"^---\s*\n(.*?)\n---", texto, re.DOTALL)
    if not match:
        return {}
    resultado = {}
    for linea in match.group(1).splitlines():
        if ":" in linea:
            clave, _, valor = linea.partition(":")
            resultado[clave.strip()] = valor.strip()
    return resultado


def verificar_conflicto(datos: dict) -> dict | None:
    fecha = datos.get("fecha", "").strip()
    hora  = datos.get("hora", "").strip()
    if not fecha or not hora or "por definir" in (fecha + hora).lower():
        return None
    try:
        h, m      = map(int, hora.split(":"))
        min_nuevo = h * 60 + m
    except ValueError:
        return None
    if not CITAS_PATH.exists():
        return None
    for archivo in CITAS_PATH.glob("*.md"):
        fm = parsear_frontmatter(archivo.read_text(encoding="utf-8"))
        if fm.get("fecha") != fecha:
            continue
        try:
            h2, m2 = map(int, fm.get("hora", "").split(":"))
            if abs(min_nuevo - h2 * 60 - m2) < 60:
                return {"fecha": fecha, "hora": fm.get("hora", hora)}
        except ValueError:
            continue
    return None


def horas_disponibles_en_fecha(fecha: str) -> list[str]:
    try:
        dia_dt     = date.fromisoformat(fecha)
        nombre_dia = DIAS_SEMANA[dia_dt.weekday()]
    except ValueError:
        return []
    rango = HORARIOS_DIA.get(nombre_dia)
    if rango is None:
        return []
    ocupados: list[int] = []
    if CITAS_PATH.exists():
        for archivo in CITAS_PATH.glob("*.md"):
            fm = parsear_frontmatter(archivo.read_text(encoding="utf-8"))
            if fm.get("fecha") != fecha:
                continue
            try:
                h, m = map(int, fm.get("hora", "").split(":"))
                ocupados.append(h * 60 + m)
            except ValueError:
                pass
    libres = []
    for slot in range(rango["inicio"], rango["fin"], 60):
        if slot + 60 > rango["fin"]:
            break
        if all(abs(slot - ocp) >= 60 for ocp in ocupados):
            libres.append(f"{slot // 60:02d}:{slot % 60:02d}")
    return libres[:8]


def calcular_proximas_disponibilidades(n_dias: int = 5) -> str:
    hoy  = date.today()
    lineas: list[str] = []
    dias_encontrados  = 0
    i = 0
    while dias_encontrados < n_dias and i < 90:
        dia   = hoy + timedelta(days=i)
        slots = horas_disponibles_en_fecha(dia.isoformat())
        if slots:
            nombre_dia = DIAS_SEMANA[dia.weekday()].capitalize()
            lineas.append(f"  • {dia.strftime('%d/%m/%Y')} ({nombre_dia}): {', '.join(slots)}")
            dias_encontrados += 1
        i += 1
    return "\n".join(lineas) if lineas else "  Sin disponibilidad próxima."


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


def guardar_cita(datos: dict) -> Path:
    CITAS_PATH.mkdir(parents=True, exist_ok=True)
    nombre  = datos.get("nombre", "por definir").replace(" ", "-")
    fecha   = datos.get("fecha", "por definir")
    hora    = datos.get("hora", "00-00").replace(":", "")
    archivo = CITAS_PATH / f"{fecha}-{hora}-{nombre}.md"
    contenido = (
        f"---\n"
        f"nombre: {datos.get('nombre', 'por definir')}\n"
        f"fecha: {datos.get('fecha', 'por definir')}\n"
        f"hora: {datos.get('hora', 'por definir')}\n"
        f"motivo: {datos.get('motivo', 'por definir')}\n"
        f"tipo: {datos.get('tipo', 'por definir')}\n"
        f"canal: whatsapp\n"
        f"---\n\n"
        f"# Cita — {datos.get('nombre', 'por definir')}\n\n"
        f"Agendada el {date.today().isoformat()} via WhatsApp (Kapso).\n"
    )
    archivo.write_text(contenido, encoding="utf-8")
    return archivo


async def push_cita_github(nombre_archivo: str, contenido: str) -> None:
    """Sube el .md de la cita al repo doctor-vault en GitHub para sincronizar con Obsidian."""
    if not GITHUB_TOKEN:
        console.print("[yellow]GITHUB_TOKEN no configurado — cita no sincronizada con Obsidian[/yellow]")
        return

    url = (
        f"https://api.github.com/repos/{GITHUB_VAULT_OWNER}/{GITHUB_VAULT_REPO}"
        f"/contents/Citas/{nombre_archivo}"
    )
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    contenido_b64 = base64.b64encode(contenido.encode("utf-8")).decode("ascii")

    # Verificar si ya existe (para obtener el sha en caso de update)
    sha = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                sha = r.json().get("sha")
    except Exception:
        pass

    payload: dict = {
        "message": f"Nueva cita: {nombre_archivo}",
        "content": contenido_b64,
    }
    if sha:
        payload["sha"] = sha

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(url, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                console.print(f"[bold green]✅ Cita sincronizada con Obsidian: {nombre_archivo}[/bold green]")
            else:
                console.print(f"[red]GitHub API error {resp.status_code}: {resp.text[:200]}[/red]")
    except Exception as e:
        console.print(f"[red]Error al subir cita a GitHub: {e}[/red]")


def extraer_datos_cita(conv: list[dict]) -> dict | None:
    texto_conv = "\n".join(
        f"{'Paciente' if m['role'] == 'user' else AGENTE_NOMBRE}: {m['content']}"
        for m in conv[-12:]
    )
    prompt = (
        f"Extrae los datos de la cita médica de esta conversación.\n"
        f"Devuelve SOLO un JSON con: nombre, fecha (YYYY-MM-DD), hora (HH:MM), motivo, tipo.\n"
        f"Si un dato no está claro, usa 'por definir'.\n"
        f"Para 'tipo' usa solo 'primera vez' o 'seguimiento'.\n\n"
        f"Conversación:\n{texto_conv}\n\nJSON:"
    )
    try:
        resp = cliente_ant.messages.create(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = resp.content[0].text
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return None


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
    if modelo_embeddings is None or collection is None:
        return "", []
    emb     = modelo_embeddings.encode([query]).tolist()
    results = collection.query(query_embeddings=emb, n_results=5)
    docs    = results.get("documents", [[]])[0]
    ids     = results.get("ids", [[]])[0]
    partes  = []
    fuentes = []
    for doc, doc_id in zip(docs, ids):
        if doc and doc.strip():
            partes.append(f"## {doc_id}\n{doc}")
            fuentes.append(doc_id)
    return "\n\n".join(partes), fuentes


# ══════════════════════════════════════════════════════════════════
#  Envío de mensajes via Kapso API
# ══════════════════════════════════════════════════════════════════

async def enviar_whatsapp(to_phone: str, texto: str) -> bool:
    """Envía un mensaje de texto via Kapso Platform API."""
    url = f"{KAPSO_API_BASE}/whatsapp/phone-numbers/{KAPSO_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {KAPSO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": to_phone,
        "type": "text",
        "text": {"body": texto},
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code not in (200, 201):
                console.print(f"[red]Kapso API error {resp.status_code}: {resp.text}[/red]")
                return False
            return True
    except Exception as e:
        console.print(f"[red]Error enviando WhatsApp: {e}[/red]")
        return False


# ══════════════════════════════════════════════════════════════════
#  Lógica principal del agente (igual que agente.py pero per-user)
# ══════════════════════════════════════════════════════════════════

async def procesar_mensaje(phone: str, mensaje: str) -> str:
    """Procesa un mensaje del paciente y devuelve la respuesta del agente."""
    historial      = historiales[phone]
    cita_pendiente = citas_pendientes[phone]

    inicio        = time.time()
    cita_guardada = None
    omitir_deteccion = False
    nota_pendiente   = ""

    # --- 1. Resolver confirmación de cita duplicada ---
    if cita_pendiente:
        respuesta_usuario = usuario_confirma(mensaje)
        if respuesta_usuario is True:
            try:
                ruta = guardar_cita(cita_pendiente)
                cita_guardada = ruta.name
                await push_cita_github(ruta.name, ruta.read_text(encoding="utf-8"))
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            nota_pendiente = (
                f"ACCIÓN COMPLETADA: El paciente confirmó que SÍ es la misma persona. "
                f"La cita de {cita_pendiente.get('nombre','?')} para el "
                f"{cita_pendiente.get('fecha','?')} a las {cita_pendiente.get('hora','?')} "
                f"ya fue guardada. Confirma con 'tu cita está registrada'.\n\n"
            )
            citas_pendientes[phone] = None
            omitir_deteccion = True
        elif respuesta_usuario is False:
            try:
                ruta = guardar_cita(cita_pendiente)
                cita_guardada = ruta.name
                await push_cita_github(ruta.name, ruta.read_text(encoding="utf-8"))
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            nota_pendiente = (
                f"ACCIÓN COMPLETADA: Es una persona diferente. Cita de "
                f"{cita_pendiente.get('nombre','?')} guardada. Ahora atiende al nuevo paciente.\n\n"
            )
            citas_pendientes[phone] = None
            omitir_deteccion = True

    # --- 2. Búsqueda RAG ---
    contexto, fuentes = buscar_contexto(mensaje)

    # --- 3. Detectar intención de agendar y extraer datos ---
    if not omitir_deteccion and detectar_confirmacion(
        historial[-1]["content"] if historial else ""
    ):
        historial.append({"role": "user", "content": mensaje})
        datos = extraer_datos_cita(historial)
        if datos and datos.get("fecha", "por definir") != "por definir":
            conflicto = verificar_conflicto(datos)
            if conflicto:
                citas_pendientes[phone] = datos
                nota_pendiente = (
                    f"ALERTA DE DUPLICADO: Ya hay una cita el {conflicto['fecha']} "
                    f"a las {conflicto['hora']}. Pregunta amablemente si es la misma persona.\n\n"
                )
            else:
                try:
                    ruta = guardar_cita(datos)
                    cita_guardada = ruta.name
                    await push_cita_github(ruta.name, ruta.read_text(encoding="utf-8"))
                    nota_pendiente = (
                        f"CITA GUARDADA EXITOSAMENTE en {ruta.name}. "
                        f"Confirma con 'tu cita está registrada' y el resumen en prosa natural.\n\n"
                    )
                except Exception as e:
                    nota_pendiente = f"Error guardando cita: {e}\n\n"
    else:
        historial.append({"role": "user", "content": mensaje})

    # Truncar historial
    if len(historial) > MAX_HISTORIAL:
        historiales[phone] = historial[-MAX_HISTORIAL:]
        historial = historiales[phone]

    # --- 4. Construir system prompt ---
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
        "• Si el paciente ya dio nombre en la conversación, NO lo pidas de nuevo.\n"
        "• Si da apellido adicional, combínalo: nombre_previo + apellido_nuevo.\n\n"

        "TIPO DE CITA:\n"
        "Si mencionó visita previa o receta → SEGUIMIENTO. Si no hay evidencia → primera vez.\n\n"

        "MOTIVO: Usa lo que el paciente describió. NUNCA inventes. Si no mencionó nada → 'por definir'.\n\n"

        "REGLAS MÉDICAS:\n"
        "1. NUNCA proporciones diagnósticos ni consejos de tratamiento.\n"
        "2. El doctor atiende CUALQUIER motivo, no solo cirugía. NUNCA redirigir a otro especialista.\n"
        "3. URGENCIAS: 'Acude a urgencias o llama al 911.'\n\n"

        "REGLA DE DÍA CERRADO: Si piden domingo → ofrece sábado/lunes con sus horas disponibles.\n\n"

        f"FECHA DE HOY: {hoy.isoformat()} ({dia_hoy}).\n"
        f"DISPONIBILIDAD PRÓXIMA:\n{calcular_proximas_disponibilidades(5)}\n\n"
        f"CALENDARIO 90 DÍAS:\n{calcular_dias_proximos()}\n\n"
        f"FECHA MÁXIMA: {fecha_maxima.isoformat()}.\n\n"

        f"{nota_pendiente}"
        "Usa solo la siguiente información del consultorio para responder:\n\n"
        f"{contexto}"
    )

    # --- 5. Llamar al LLM ---
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

    ms = int((time.time() - inicio) * 1000)
    console.print(f"[dim]{phone} → {ms}ms | fuentes: {fuentes}[/dim]")
    return texto_respuesta


# ══════════════════════════════════════════════════════════════════
#  Endpoints FastAPI
# ══════════════════════════════════════════════════════════════════

def verificar_firma_kapso(body: bytes, signature: str) -> bool:
    """Verifica la firma HMAC del webhook de Kapso (opcional pero recomendado)."""
    if not KAPSO_WEBHOOK_SECRET:
        return True
    expected = hmac.new(
        KAPSO_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.get("/webhook")
async def webhook_verify(request: Request):
    """Verificación de webhook (algunos proveedores envían GET para validar)."""
    params = dict(request.query_params)
    challenge = params.get("hub.challenge") or params.get("challenge")
    if challenge:
        return JSONResponse(content={"challenge": challenge})
    return JSONResponse(content={"status": "ok"})


@app.post("/webhook")
async def webhook_recibir(request: Request):
    """
    Recibe eventos de Kapso (whatsapp.message.received) y responde al usuario.

    Estructura esperada del payload de Kapso:
    {
      "event": "whatsapp.message.received",
      "data": {
        "from": "521234567890",
        "text": { "body": "Hola" },
        "type": "text"
      }
    }
    """
    body_bytes = await request.body()

    # Verificar firma (si está configurada)
    sig = request.headers.get("X-Kapso-Signature", "")
    if not verificar_firma_kapso(body_bytes, sig):
        raise HTTPException(status_code=401, detail="Firma inválida")

    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido")

    evento = payload.get("event", "")

    # Solo procesar mensajes de texto entrantes
    if evento != "whatsapp.message.received":
        return JSONResponse(content={"status": "ignored"})

    data = payload.get("data", {})
    tipo = data.get("type", "")

    if tipo != "text":
        # Ignorar imágenes, audio, etc. (puedes expandir esto)
        return JSONResponse(content={"status": "non-text ignored"})

    phone   = data.get("from", "")
    mensaje = data.get("text", {}).get("body", "").strip()

    if not phone or not mensaje:
        return JSONResponse(content={"status": "empty"})

    console.print(f"[cyan]📱 {phone}:[/cyan] {mensaje[:80]}")

    # Procesar con el agente
    respuesta = await procesar_mensaje(phone, mensaje)

    # Enviar respuesta via Kapso
    await enviar_whatsapp(phone, respuesta)

    return JSONResponse(content={"status": "ok"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "negocio": NEGOCIO_NOMBRE,
        "agente": AGENTE_NOMBRE,
        "usuarios_activos": len(historiales),
    }


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    console.print(f"[bold green]🚀 {AGENTE_NOMBRE} — {NEGOCIO_NOMBRE} vía Kapso/WhatsApp[/bold green]")
    console.print(f"[yellow]Apunta el webhook de Kapso a: POST /webhook (puerto {port})[/yellow]")
    uvicorn.run("agente_kapso:app", host="0.0.0.0", port=port, reload=False)
