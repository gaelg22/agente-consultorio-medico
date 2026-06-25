import json
import os
import re
import sys
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import anthropic
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from sentence_transformers import SentenceTransformer

console = Console()

VAULT_PATH  = Path(r"C:\Users\gaelg_12gxbrs\OneDrive\Documentos\Doctor-Agente")
CITAS_PATH  = VAULT_PATH / "Citas"
CHROMA_PATH = Path("./chroma_db")

# ─── Palabras clave ────────────────────────────────────────────────────────────

PALABRAS_PRECIOS = [
    "precio", "costo", "cuanto", "cuánto", "cuanto cuesta", "cuánto cuesta",
    "cuanto cobra", "cuánto cobra", "cuanto vale", "cuánto vale",
    "tarifa", "cobran", "honorarios",
]

PALABRAS_CIRUGIA = [
    "cirugia", "cirugía", "operacion", "operación", "operar", "quirúrgico",
    "quirurgico", "laparoscopica", "laparoscópica", "hernia", "vesicula",
    "vesícula", "apendice", "apéndice",
]

PALABRAS_PREPARACION = [
    "preparacion", "preparación", "preparar", "ayuno", "preoperatorio",
    "antes de la cirugia", "antes de la cirugía", "lista", "que traer",
    "qué traer", "indicaciones",
]

FRASES_CONFIRMACION = [
    "tu cita está registrada",
    "tu cita queda registrada",
    "quedas agendado",
    "quedas agendada",
    "cita confirmada",
    "ya quedaste agendado",
    "ya quedaste agendada",
    "tu cita queda",
    "te registramos",
    "te agendamos",
]

PALABRAS_AFIRMACION = [
    "si", "sí", "soy yo", "ese soy", "esa soy", "correcto",
    "exacto", "claro que si", "soy el mismo", "soy la misma", "asi es",
]
PALABRAS_NEGACION = [
    "no", "otra persona", "somos diferentes", "no soy yo",
    "diferente", "no es", "no, soy",
]

DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Horarios por día de semana (índice 0=lunes ... 6=domingo)
HORARIOS_DIA = {
    "lunes":     {"inicio": 9 * 60,  "fin": 14 * 60},   # 9:00 – 14:00
    "martes":    {"inicio": 16 * 60, "fin": 20 * 60},   # 16:00 – 20:00
    "miercoles": {"inicio": 9 * 60,  "fin": 14 * 60},   # 9:00 – 14:00
    "jueves":    {"inicio": 16 * 60, "fin": 20 * 60},   # 16:00 – 20:00
    "viernes":   {"inicio": 9 * 60,  "fin": 14 * 60},   # 9:00 – 14:00
    "sabado":    {"inicio": 9 * 60,  "fin": 13 * 60},   # 9:00 – 13:00
    "domingo":   None,                                   # CERRADO
}

historial: list[dict] = []
cita_pendiente: dict | None = None
model_st: SentenceTransformer | None = None
collection = None
cliente_ant: anthropic.Anthropic | None = None


def setup_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        console.print("[green]API key cargada desde .env[/green]")
        return api_key

    console.print("[yellow]No se encontro API key de Anthropic.[/yellow]")
    while True:
        api_key = input("Pega tu API key de Anthropic (empieza con sk-ant-): ").strip()
        if api_key.startswith("sk-ant-"):
            Path(".env").write_text(f"ANTHROPIC_API_KEY={api_key}\n", encoding="utf-8")
            os.environ["ANTHROPIC_API_KEY"] = api_key
            console.print("[green]API key guardada en .env[/green]")
            return api_key
        console.print("[red]Formato invalido. Debe empezar con 'sk-ant-'. Intenta de nuevo.[/red]")


def buscar_cita_existente(nombre: str) -> dict | None:
    if not CITAS_PATH.exists():
        return None
    clave = re.sub(r"[^\w\s-]", "", nombre).strip().lower().replace(" ", "-")
    for archivo in sorted(CITAS_PATH.glob("*.md"), reverse=True):
        if clave in archivo.stem.lower():
            partes = archivo.stem.split("-")
            if len(partes) >= 4:
                fecha    = f"{partes[0]}-{partes[1]}-{partes[2]}"
                hora_raw = partes[3]
                hora     = f"{hora_raw[:2]}:{hora_raw[2:]}" if len(hora_raw) >= 4 else hora_raw
                return {"fecha": fecha, "hora": hora, "archivo": archivo.name}
    return None


def usuario_confirma(mensaje: str) -> bool | None:
    m = mensaje.lower().strip()
    if any(p in m for p in PALABRAS_NEGACION):
        return False
    if any(p in m for p in PALABRAS_AFIRMACION):
        return True
    return None


def parsear_frontmatter(contenido: str) -> dict:
    if not contenido.startswith("---"):
        return {}
    partes = contenido.split("---", 2)
    if len(partes) < 3:
        return {}
    resultado = {}
    for linea in partes[1].strip().split("\n"):
        if ":" in linea:
            clave, _, valor = linea.partition(":")
            resultado[clave.strip()] = valor.strip()
    return resultado


def verificar_conflicto(datos: dict) -> dict | None:
    """Verifica si ya existe una cita en la misma fecha y hora (dentro de 60 min)."""
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
    """Devuelve lista de horas libres en la fecha indicada según horario del día.
    Intervalos de 1 hora. El último slot debe caber completo antes del cierre."""
    try:
        dia_dt     = date.fromisoformat(fecha)
        nombre_dia = DIAS_SEMANA[dia_dt.weekday()]
    except ValueError:
        return []

    rango = HORARIOS_DIA.get(nombre_dia)
    if rango is None:
        return []  # Domingo — cerrado

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
        # El slot debe caber completo (1 hora) antes del cierre
        if slot + 60 > rango["fin"]:
            break
        if all(abs(slot - ocp) >= 60 for ocp in ocupados):
            libres.append(f"{slot // 60:02d}:{slot % 60:02d}")
    return libres[:8]


def calcular_proximas_disponibilidades(n_dias: int = 5) -> str:
    """Devuelve los próximos N días con slots disponibles y sus horas."""
    hoy   = date.today()
    lineas: list[str] = []
    dias_encontrados  = 0
    i = 0
    while dias_encontrados < n_dias and i < 90:
        dia    = hoy + timedelta(days=i)
        slots  = horas_disponibles_en_fecha(dia.isoformat())
        if slots:
            nombre_dia = DIAS_SEMANA[dia.weekday()].capitalize()
            lineas.append(f"  • {dia.strftime('%d/%m/%Y')} ({nombre_dia}): {', '.join(slots)}")
            dias_encontrados += 1
        i += 1
    return "\n".join(lineas) if lineas else "  Sin disponibilidad próxima."


def detectar_confirmacion(texto: str) -> bool:
    texto_lower = texto.lower()
    return any(frase in texto_lower for frase in FRASES_CONFIRMACION)


def extraer_datos_cita(conv: list[dict]) -> dict | None:
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in conv[-12:]
    )

    hoy        = date.today()
    dia_semana = DIAS_SEMANA[hoy.weekday()]

    resp = cliente_ant.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Analiza esta conversacion de un consultorio medico y extrae los datos de la cita confirmada.\n"
                f"HOY es {hoy.isoformat()} ({dia_semana}). Usa esta fecha para calcular referencias "
                "relativas como 'manana', 'pasado manana', 'el viernes', 'el lunes', etc.\n"
                "Responde UNICAMENTE con un JSON valido, sin texto adicional, con esta estructura:\n"
                '{"nombre": "Nombre Apellido", "fecha": "YYYY-MM-DD", "hora": "HH:MM", '
                '"motivo": "...", "tipo": "primera vez|seguimiento"}\n'
                "El campo 'nombre' debe incluir nombre Y apellido si ambos fueron mencionados. "
                "La fecha SIEMPRE debe estar en formato YYYY-MM-DD. "
                "El campo 'motivo' es la razon breve de la consulta. "
                "El campo 'tipo' indica si es primera vez o seguimiento. "
                "Si algun dato no se menciono, usa \"por definir\".\n\n"
                f"CONVERSACION:\n{conv_text}"
            ),
        }],
    )

    raw   = resp.content[0].text.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def guardar_cita(datos: dict) -> Path:
    CITAS_PATH.mkdir(parents=True, exist_ok=True)

    fecha    = datos.get("fecha", "0000-00-00")
    hora     = datos.get("hora", "00:00")
    nombre   = datos.get("nombre", "paciente")
    motivo   = datos.get("motivo", "por definir")
    tipo     = datos.get("tipo", "por definir")

    hora_archivo   = hora.replace(":", "")[:4]
    nombre_limpio  = re.sub(r"[^\w\s-]", "", nombre).strip().replace(" ", "-")
    nombre_archivo = f"{fecha}-{hora_archivo}-{nombre_limpio}.md"
    ruta           = CITAS_PATH / nombre_archivo

    contenido = (
        f"---\n"
        f"tags: cita\n"
        f"fecha: {fecha}\n"
        f"hora: {hora}\n"
        f"motivo: {motivo}\n"
        f"tipo: {tipo}\n"
        f"estado: pendiente\n"
        f"---\n\n"
        f"# Cita — {nombre}\n\n"
        f"- **Fecha:** {fecha}\n"
        f"- **Hora:** {hora}\n"
        f"- **Motivo:** {motivo}\n"
        f"- **Tipo:** {tipo}\n"
        f"- **Estado:** pendiente\n\n"
        f"## Recordatorios\n"
        f"- Traer estudios previos y resultados de laboratorio\n"
        f"- Llegar 10 minutos antes\n"
        f"- Llevar lista de medicamentos actuales\n\n"
        f"## Relacionado\n"
        f"[[identidad]] [[informacion-consulta]] [[horarios-politicas]]\n"
    )

    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def indexar_nota(ruta: Path) -> None:
    content   = ruta.read_text(encoding="utf-8")
    embedding = model_st.encode(content).tolist()
    collection.upsert(
        documents=[content],
        embeddings=[embedding],
        ids=[ruta.stem],
        metadatas=[{"nombre": ruta.stem, "archivo": ruta.name}],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_st, collection, cliente_ant

    api_key = setup_api_key()

    if not CHROMA_PATH.exists():
        console.print("[yellow]No existe indice ChromaDB. Indexando vault de Obsidian...[/yellow]")
        subprocess.run([sys.executable, "indexar.py"], check=True)

    console.print("[cyan]Cargando modelo de embeddings...[/cyan]")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection    = chroma_client.get_collection("doctor")

    cliente_ant = anthropic.Anthropic(api_key=api_key)

    console.print("[bold green]Asistente del Dr. Alejandro Martínez listo en http://localhost:8001[/bold green]")
    yield


app = FastAPI(title="Asistente del Dr. Alejandro Martínez — Cirugía General", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=Path("index.html").read_text(encoding="utf-8"))


def calcular_dias_proximos() -> str:
    """Genera un calendario de los proximos 90 dias para que Claude pueda verificar fechas."""
    hoy   = date.today()
    lineas = []
    for i in range(90):
        dia        = hoy + timedelta(days=i)
        nombre_dia = DIAS_SEMANA[dia.weekday()]
        horario    = HORARIOS_DIA.get(nombre_dia)
        if horario is None:
            estado = "CERRADO"
        else:
            h_ini = horario["inicio"] // 60
            m_ini = horario["inicio"] % 60
            h_fin = horario["fin"] // 60
            m_fin = horario["fin"] % 60
            estado = f"{h_ini:02d}:{m_ini:02d}–{h_fin:02d}:{m_fin:02d}"
        lineas.append(f"{dia.isoformat()} = {nombre_dia} ({estado})")
    return "\n".join(lineas)


@app.post("/chat")
async def chat(req: ChatRequest):
    global historial, cita_pendiente

    inicio        = time.time()
    cita_guardada: str | None = None
    omitir_deteccion = False
    nota_pendiente   = ""

    # --- 1. Resolver confirmacion de cita duplicada (turno previo) ---
    if cita_pendiente:
        respuesta_usuario = usuario_confirma(req.message)

        if respuesta_usuario is True:
            try:
                ruta          = guardar_cita(cita_pendiente)
                indexar_nota(ruta)
                cita_guardada = ruta.name
                console.print(f"[bold green]Cita guardada (confirmacion positiva): {ruta.name}[/bold green]")
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            nota_pendiente = (
                f"ACCION COMPLETADA: El paciente confirmo que SI es la misma persona. "
                f"La cita de {cita_pendiente.get('nombre','?')} para el {cita_pendiente.get('fecha','?')} "
                f"a las {cita_pendiente.get('hora','?')} "
                f"(motivo: {cita_pendiente.get('motivo','?')}) ya fue guardada en el sistema. "
                "Confirma al paciente usando la frase 'tu cita está registrada'.\n\n"
            )
            cita_pendiente   = None
            omitir_deteccion = True

        elif respuesta_usuario is False:
            try:
                ruta          = guardar_cita(cita_pendiente)
                indexar_nota(ruta)
                cita_guardada = ruta.name
                console.print(f"[bold green]Cita guardada (persona diferente): {ruta.name}[/bold green]")
            except Exception as e:
                console.print(f"[red]Error guardando cita: {e}[/red]")
            nota_pendiente = (
                f"ACCION COMPLETADA: El paciente indico que NO es la misma persona de la cita anterior. "
                f"Se registro la nueva cita a nombre de {cita_pendiente.get('nombre','?')} para el "
                f"{cita_pendiente.get('fecha','?')} a las {cita_pendiente.get('hora','?')} "
                f"(motivo: {cita_pendiente.get('motivo','?')}). "
                "Confirma la nueva cita al paciente usando la frase 'tu cita está registrada'.\n\n"
            )
            cita_pendiente   = None
            omitir_deteccion = True

        else:
            nota_pendiente = (
                f"PENDIENTE: No quedo claro si el paciente actual es la misma persona de una cita previa "
                f"a nombre de {cita_pendiente.get('nombre','?')} para el {cita_pendiente.get('fecha','?')} "
                f"a las {cita_pendiente.get('hora','?')}. "
                "Pregunta de nuevo con amabilidad: '¿Es usted la misma persona o desea registrar una cita nueva?'\n\n"
            )

    # --- 2. RAG: buscar contexto en ChromaDB ---
    query_embedding = model_st.encode(req.message).tolist()
    resultados      = collection.query(query_embeddings=[query_embedding], n_results=5)

    fuentes: list[str]       = []
    partes_contexto: list[str] = []
    for i, doc in enumerate(resultados["documents"][0]):
        nombre = resultados["metadatas"][0][i]["nombre"]
        fuentes.append(nombre)
        partes_contexto.append(f"## {nombre}\n{doc}")

    mensaje_lower = req.message.lower()

    # Forzar inclusion de servicios-consultas.md cuando preguntan precios
    if any(p in mensaje_lower for p in PALABRAS_PRECIOS) and "servicios-consultas" not in fuentes:
        precio_results = collection.get(ids=["servicios-consultas"])
        if precio_results["documents"]:
            fuentes.append("servicios-consultas")
            partes_contexto.append(f"## servicios-consultas\n{precio_results['documents'][0]}")

    # Forzar inclusion de preparacion-cirugia.md cuando preguntan de cirugias/preparacion
    if any(p in mensaje_lower for p in PALABRAS_CIRUGIA + PALABRAS_PREPARACION) and "preparacion-cirugia" not in fuentes:
        prep_results = collection.get(ids=["preparacion-cirugia"])
        if prep_results["documents"]:
            fuentes.append("preparacion-cirugia")
            partes_contexto.append(f"## preparacion-cirugia\n{prep_results['documents'][0]}")

    contexto = "\n\n---\n\n".join(partes_contexto)

    hoy          = date.today()
    dia_hoy      = DIAS_SEMANA[hoy.weekday()]
    fecha_maxima = hoy + timedelta(days=90)

    system_prompt = (
        "Eres Mateo, parte del equipo del consultorio del Dr. Alejandro Martínez (Cirugía General).\n"
        "Hablas en español, con un tono tranquilo, cálido, cercano y empático.\n"
        "Nunca dices que eres una IA ni un asistente virtual. Eres Mateo, del equipo.\n\n"

        "⚠️ REGLA ABSOLUTA — NUNCA TE PRESENTES NI DIGAS TU NOMBRE:\n"
        "Ya existe un mensaje de bienvenida visible para el paciente donde ya te presentaste.\n"
        "PROHIBIDO repetir frases como 'Soy Mateo', 'Me llamo Mateo', 'Soy del equipo', etc.\n"
        "Si el paciente dice 'Hola' o cualquier saludo, responde con calidez pero SIN decir tu nombre.\n"
        "Ejemplo CORRECTO si dicen 'Hola': '¡Qué gusto! ¿En qué te puedo ayudar?'\n"
        "Ejemplo INCORRECTO: '¡Hola! Soy Mateo, del equipo del Dr. Alejandro Martínez...'\n\n"

        "═══ PERFIL DEL PACIENTE ═══\n"
        "Los pacientes de este consultorio tienden a ser ansiosos o hipocondríacos: se preocupan mucho por sus síntomas.\n"
        "Tu prioridad es TRANQUILIZARLOS antes de hacer cualquier otra cosa.\n"
        "NO hagas muchas preguntas seguidas. Haz UNA sola pregunta a la vez, con calma.\n"
        "No seas entrometido ni clínico. Habla con naturalidad, como si fuera una plática amigable.\n\n"
        "⚠️ FRASES PROHIBIDAS — NUNCA uses estas expresiones:\n"
        "PROHIBIDO: 'Te entiendo', 'Entiendo perfectamente', 'Entiendo tu preocupación', 'Entiendo cómo te sientes'.\n"
        "Razón: no podemos realmente conocer el dolor o sentimiento del paciente, y esas frases pueden sonar falsas o condescendientes.\n"
        "En su lugar usa frases como:\n"
        "  ✓ 'Qué bueno que nos contactas'\n"
        "  ✓ 'Es válido que te preocupe'\n"
        "  ✓ 'Eso que describes suena incómodo'\n"
        "  ✓ 'Gracias por contarnos'\n"
        "  ✓ 'Es normal que eso genere dudas'\n"
        "  ✓ 'Qué difícil que siga el dolor'\n\n"

        "═══ FLUJO DE ATENCIÓN — RÁPIDO Y SIN RETENER AL PACIENTE ═══\n"
        "Regla general: sé breve y conversacional. Habla como una persona, no como un formulario.\n\n"

        "⚡ MODO AGENDAMIENTO DIRECTO — MÁXIMA PRIORIDAD:\n"
        "Si el paciente pide una cita directamente ('quiero agendar', 'necesito una cita', 'apártame una cita'), "
        "activa el MODO AGENDAMIENTO. En este modo:\n"
        "  - Ve directo a mostrar los horarios disponibles.\n"
        "  - Una vez elegido el horario, pide SOLO lo que falte: nombre (si no lo dio) y motivo (si no lo mencionó).\n"
        "  - NUNCA preguntes si tiene tratamiento ni hagas el flujo de síntomas. El paciente ya decidió venir.\n"
        "  - Si el paciente menciona su motivo durante el agendamiento, acéptalo y confirma la cita. "
        "NO respondas con '¿Ya tienes algo indicado?' — ESO ESTÁ PROHIBIDO en modo agendamiento.\n"
        "  - El tipo de cita es 'primera vez' si no hay evidencia de visita previa, 'seguimiento' si la hay.\n\n"

        "FLUJO DE SÍNTOMAS (solo cuando el paciente reporta una molestia SIN pedir cita):\n"
        "PASO 1+2: En UN solo mensaje, valida brevemente y pregunta si ya tiene tratamiento indicado.\n"
        "PASO 3: Si tiene tratamiento y está estable, tranquilízalo y cierra. Si no mejora o no tiene tratamiento,\n"
        "recomienda la cita Y muestra los horarios disponibles en ESE MISMO mensaje. No preguntes '¿quieres una cita?'.\n\n"

        "TONO Y ESTILO DE RESPUESTA:\n"
        "Escribe como si fuera una conversación natural, no un formulario médico.\n"
        "PROHIBIDO usar listas con bullets o campos etiquetados (Nombre:, Fecha:, Motivo:) en las confirmaciones.\n"
        "La confirmación debe sonar como una persona real hablando. Ejemplo correcto:\n"
        "  'Listo, Luis Mario. Tu cita está registrada para el miércoles 24 a las 11:00, "
        "ya queda anotado. Llega unos 10 minutos antes y si tienes estudios o análisis previos, tráelos.'\n"
        "La frase exacta 'tu cita está registrada' debe aparecer siempre, pero integrada en texto natural.\n\n"

        "RECONOCIMIENTO DEL NOMBRE:\n"
        "• Extrae el nombre ignorando artículos (el, la, los). 'Soy el Luis' → nombre = 'Luis'.\n"
        "• Dos o más palabras = nombre completo (puede incluir apellido). Solo pide apellido si dio UNA sola palabra.\n"
        "• Si el paciente dice 'Ya te lo di', búscalo en el historial y úsalo sin pedirlo de nuevo.\n"
        "• Si ya dio nombre (ej: 'Cristiano Ronaldo') y luego da apellido ('Torres'), combínalos: 'Cristiano Ronaldo Torres'.\n\n"

        "RECONOCIMIENTO DEL TIPO DE CITA:\n"
        "Si mencionó visita previa, receta o seguimiento en algún momento → SEGUIMIENTO. Si no hay evidencia → primera vez.\n"
        "NUNCA preguntes el tipo si ya se puede deducir.\n\n"

        "MOTIVO:\n"
        "Usa lo que el paciente describió. NUNCA inventes. Si no mencionó nada → 'por definir'.\n\n"

        "Pide solo datos realmente faltantes. Combina varios en UN mensaje si hace falta.\n"


        "═══ REGLAS MÉDICAS OBLIGATORIAS ═══\n"
        "1. NUNCA proporciones diagnósticos médicos, interpretaciones de estudios ni consejos de tratamiento.\n"
        "2. NUNCA recomiendes ni ajustes medicamentos.\n"
        "3. Si el paciente describe síntomas, tranquilízalo primero (PASO 1 y 2) antes de hablar de citas.\n"
        "4. CONSULTA GENERAL: El Dr. Alejandro Martínez atiende CUALQUIER motivo de consulta, no solo cirugía. "
        "NUNCA digas que el paciente debería ir con otro especialista ni que el doctor no maneja ese padecimiento. "
        "Para cualquier molestia —pie, cabeza, espalda, lo que sea— sigue el flujo normal: calmar → tratamiento → cita.\n"
        "5. URGENCIAS MÉDICAS: Si el paciente describe dolor severo, sangrado abundante, fiebre alta, "
        "dificultad para respirar u otra emergencia, responde INMEDIATAMENTE:\n"
        "   'Por favor, acude de inmediato a urgencias o llama al 911. Este consultorio no atiende urgencias.'\n\n"

        f"FECHA DE HOY: {hoy.isoformat()} ({dia_hoy}).\n"
        f"DISPONIBILIDAD PRÓXIMA (próximos días con horas libres):\n"
        f"{calcular_proximas_disponibilidades(5)}\n\n"
        f"CALENDARIO PROXIMOS 90 DIAS (horario por día, para verificar día de semana):\n"
        f"{calcular_dias_proximos()}\n\n"
        f"FECHA MAXIMA PARA AGENDAR: {fecha_maxima.isoformat()} (90 días a partir de hoy).\n"
        "REGLA DE FECHAS: Si el paciente pide una fecha posterior a la fecha máxima, "
        "recházala amablemente y ofrece la fecha más lejana disponible.\n\n"

        "REGLA DE CALENDARIO: Antes de confirmar cualquier cita, SIEMPRE verifica el día de semana "
        "real de la fecha solicitada usando el calendario de arriba. Si el paciente dice 'el viernes 10' "
        "pero el 10 no es viernes, corrígelo amablemente. Nunca agendes en fecha con día incorrecto.\n"
        "REGLA DE DÍA CERRADO: Si el paciente pide domingo u otro día sin atención, ofrece la alternativa "
        "MÁS CERCANA (sábado anterior o lunes siguiente) e incluye de inmediato sus horas disponibles. "
        "NUNCA digas solo 'tenemos disponibilidad el sábado' sin mostrar las horas. "
        "Ejemplo correcto: 'El domingo estamos cerrados, pero el sábado 27 tenemos: 09:00, 10:00, 11:00, 12:00. "
        "¿Alguna te viene bien?'\n\n"

        "HORARIOS POR DÍA:\n"
        "  - Lunes, Miércoles, Viernes: 9:00 am – 2:00 pm\n"
        "  - Martes, Jueves: 4:00 pm – 8:00 pm\n"
        "  - Sábado: 9:00 am – 1:00 pm\n"
        "  - Domingo: CERRADO\n"
        "Si el paciente pide domingo, ofrece el sábado anterior o el lunes siguiente.\n\n"

        f"{nota_pendiente}"
        "Usa exclusivamente la siguiente información del consultorio para responder. "
        "No inventes precios, servicios ni procedimientos:\n\n"
        f"{contexto}"
    )

    historial.append({"role": "user", "content": req.message})
    if len(historial) > 20:
        historial = historial[-20:]

    respuesta_ant = cliente_ant.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=historial,  # historial completo (máx 20 turnos)
    )

    texto = respuesta_ant.content[0].text
    historial.append({"role": "assistant", "content": texto})

    # --- 3. Post-generacion: detectar confirmacion y verificar duplicados ---
    if not omitir_deteccion and detectar_confirmacion(texto):
        console.print("[yellow]Confirmacion de cita detectada. Extrayendo datos...[/yellow]")
        datos = extraer_datos_cita(historial)
        if datos:
            existente = buscar_cita_existente(datos.get("nombre", ""))
            if existente:
                cita_pendiente = datos
                nombre_dup     = datos.get("nombre", "?")
                texto_pregunta = (
                    f"Tengo registrada una cita a nombre de {nombre_dup} "
                    f"para el {existente['fecha']} a las {existente['hora']}. "
                    "¿Es usted la misma persona o desea registrar una cita nueva?"
                )
                historial[-1]["content"] = texto_pregunta
                texto = texto_pregunta
                console.print(f"[yellow]Duplicado para '{nombre_dup}'. Solicitando confirmacion.[/yellow]")
            else:
                conflicto = verificar_conflicto(datos)
                if conflicto:
                    fecha_conf = datos.get("fecha", "?")
                    hora_conf  = datos.get("hora", "?")
                    libres     = horas_disponibles_en_fecha(fecha_conf)

                    partes_msg = [
                        f"Lo siento, ya hay una cita registrada a las {conflicto['hora']} en esa fecha."
                    ]
                    if libres:
                        partes_msg.append(f"Horas disponibles ese día: {', '.join(libres)}.")
                    partes_msg.append("¿Alguna de esas horas le acomoda?")

                    texto_conflicto          = " ".join(partes_msg)
                    historial[-1]["content"] = texto_conflicto
                    texto                    = texto_conflicto
                    console.print(f"[red]Conflicto el {fecha_conf} a las {hora_conf}.[/red]")
                else:
                    try:
                        ruta          = guardar_cita(datos)
                        indexar_nota(ruta)
                        cita_guardada = ruta.name
                        console.print(f"[bold green]Cita guardada y indexada: {ruta.name}[/bold green]")
                    except Exception as e:
                        console.print(f"[red]Error guardando cita: {e}[/red]")
        else:
            console.print("[red]No se pudieron extraer los datos de la cita.[/red]")

    ms = int((time.time() - inicio) * 1000)
    console.print(f"\n[dim]Pregunta:[/dim] {req.message}")
    console.print(f"[blue]Fuentes usadas:[/blue] {', '.join(fuentes)}")
    console.print(f"[dim]Tiempo: {ms}ms[/dim]")

    return {"response": texto, "fuentes": fuentes, "ms": ms, "cita_guardada": cita_guardada}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
