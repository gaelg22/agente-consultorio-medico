"""
db_mongo.py — Cliente MongoDB para citas del consultorio Dr. Alejandro Martínez
Usa Motor (async driver de MongoDB) para operaciones no bloqueantes.
"""

import os
from datetime import date, datetime, timedelta
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from rich.console import Console

console = Console()

# ── Configuración ──────────────────────────────────────────────────
MONGODB_URI  = os.getenv("MONGODB_URI", "")
DB_NAME      = os.getenv("MONGODB_DB", "consultorio")
COLLECTION   = "citas"

# Horarios por día (mismo que el agente)
HORARIOS_DIA = {
    "lunes":     {"inicio": 9 * 60,  "fin": 14 * 60},
    "martes":    {"inicio": 16 * 60, "fin": 20 * 60},
    "miercoles": {"inicio": 9 * 60,  "fin": 14 * 60},
    "jueves":    {"inicio": 16 * 60, "fin": 20 * 60},
    "viernes":   {"inicio": 9 * 60,  "fin": 14 * 60},
    "sabado":    {"inicio": 9 * 60,  "fin": 13 * 60},
    "domingo":   None,
}
DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# ── Cliente global ─────────────────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None


def get_db():
    """Devuelve la colección de citas."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client[DB_NAME][COLLECTION]


async def init_mongo():
    """Inicializa la conexión y crea índices."""
    db = get_db()
    await db.create_index([("fecha", 1), ("hora", 1)])
    await db.create_index([("nombre", 1)])
    await db.create_index([("telefono", 1)])
    console.print("[bold green]✅ MongoDB conectado[/bold green]")


# ── CRUD de citas ──────────────────────────────────────────────────

async def guardar_cita(datos: dict, telefono: str = "") -> dict:
    """Inserta una nueva cita en MongoDB. Devuelve el documento insertado."""
    db = get_db()
    doc = {
        "nombre":    datos.get("nombre", "por definir"),
        "telefono":  telefono,
        "fecha":     datos.get("fecha", "por definir"),
        "hora":      datos.get("hora", "por definir"),
        "motivo":    datos.get("motivo", "por definir"),
        "tipo":      datos.get("tipo", "primera vez"),
        "canal":     datos.get("canal", "whatsapp"),
        "estado":    "pendiente",
        "creada_en": datetime.utcnow(),
    }
    result = await db.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    console.print(f"[bold green]✅ Cita guardada en MongoDB: {doc['nombre']} — {doc['fecha']} {doc['hora']}[/bold green]")
    return doc


async def buscar_cita_existente(nombre: str) -> Optional[dict]:
    """Busca si ya existe una cita a nombre de ese paciente (últimos 90 días)."""
    db = get_db()
    hoy       = date.today()
    hace_90   = (hoy - timedelta(days=90)).isoformat()
    nombre_lc = nombre.lower().strip()

    async for cita in db.find({"fecha": {"$gte": hace_90}}).sort("fecha", -1):
        if nombre_lc in cita.get("nombre", "").lower():
            return cita
    return None


async def verificar_conflicto(fecha: str, hora: str) -> Optional[dict]:
    """Verifica si hay una cita existente en la misma fecha/hora (dentro de 60 min)."""
    db = get_db()
    if not fecha or not hora or "por definir" in (fecha + hora).lower():
        return None

    try:
        h, m       = map(int, hora.split(":"))
        min_nuevo  = h * 60 + m
    except ValueError:
        return None

    async for cita in db.find({"fecha": fecha, "estado": {"$ne": "cancelada"}}):
        try:
            h2, m2 = map(int, cita["hora"].split(":"))
            if abs(min_nuevo - h2 * 60 - m2) < 60:
                return cita
        except (ValueError, KeyError):
            continue
    return None


async def horas_disponibles(fecha: str) -> list[str]:
    """Devuelve horas libres en la fecha indicada según horario del día."""
    try:
        dia_dt     = date.fromisoformat(fecha)
        nombre_dia = DIAS_SEMANA[dia_dt.weekday()]
    except ValueError:
        return []

    rango = HORARIOS_DIA.get(nombre_dia)
    if rango is None:
        return []

    # Obtener horas ocupadas desde MongoDB
    db      = get_db()
    ocupados: list[int] = []
    async for cita in db.find({"fecha": fecha, "estado": {"$ne": "cancelada"}}):
        try:
            h, m = map(int, cita["hora"].split(":"))
            ocupados.append(h * 60 + m)
        except (ValueError, KeyError):
            pass

    # Minutos actuales (para filtrar slots pasados si es hoy)
    from datetime import datetime as dt
    es_hoy = (dia_dt == date.today())
    ahora_min = dt.now().hour * 60 + dt.now().minute if es_hoy else 0

    libres = []
    for slot in range(rango["inicio"], rango["fin"], 60):
        if slot + 60 > rango["fin"]:
            break
        # Omitir slots que ya pasaron hoy (con 30 min de margen mínimo)
        if es_hoy and slot <= ahora_min + 30:
            continue
        if all(abs(slot - ocp) >= 60 for ocp in ocupados):
            libres.append(f"{slot // 60:02d}:{slot % 60:02d}")
    return libres[:8]


async def proximas_disponibilidades(n_dias: int = 5) -> str:
    """Devuelve los próximos N días con slots disponibles y sus horas."""
    hoy  = date.today()
    lineas: list[str] = []
    dias_encontrados  = 0
    i = 0
    while dias_encontrados < n_dias and i < 90:
        dia   = hoy + timedelta(days=i)
        slots = await horas_disponibles(dia.isoformat())
        if slots:
            nombre_dia = DIAS_SEMANA[dia.weekday()].capitalize()
            lineas.append(f"  • {dia.strftime('%d/%m/%Y')} ({nombre_dia}): {', '.join(slots)}")
            dias_encontrados += 1
        i += 1
    return "\n".join(lineas) if lineas else "  Sin disponibilidad próxima."


async def get_citas_recientes(limit: int = 50) -> list[dict]:
    """Para el panel de admin: últimas citas."""
    db    = get_db()
    citas = []
    async for c in db.find().sort("creada_en", -1).limit(limit):
        c["_id"] = str(c["_id"])
        citas.append(c)
    return citas
