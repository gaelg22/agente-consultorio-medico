"""
db_postgres.py — Cliente PostgreSQL para logs de conversaciones y feedback del doctor.
Usa asyncpg para operaciones no bloqueantes.
"""

import os
from datetime import datetime
from typing import Optional
import asyncpg
from rich.console import Console

console = Console()

POSTGRES_URL = os.getenv("POSTGRES_URL", "")

_pool: Optional[asyncpg.Pool] = None


async def init_postgres():
    """Crea el pool de conexiones y las tablas si no existen."""
    global _pool
    _pool = await asyncpg.create_pool(POSTGRES_URL, min_size=1, max_size=5)

    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                rol         TEXT NOT NULL CHECK (rol IN ('user', 'assistant')),
                mensaje     TEXT NOT NULL,
                timestamp   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id               SERIAL PRIMARY KEY,
                session_id       TEXT,
                respuesta_mala   TEXT NOT NULL,
                contexto         TEXT,
                correccion       TEXT NOT NULL,
                nota_generada    TEXT,
                procesado_en     TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversaciones(session_id);
            CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversaciones(timestamp DESC);
        """)
    console.print("[bold green]✅ PostgreSQL conectado y tablas listas[/bold green]")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("PostgreSQL no inicializado. Llama a init_postgres() primero.")
    return _pool


# ── Logs de conversación ───────────────────────────────────────────

async def log_mensaje(session_id: str, rol: str, mensaje: str) -> int:
    """Guarda un mensaje en el log de conversaciones. Devuelve el ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO conversaciones (session_id, rol, mensaje) VALUES ($1, $2, $3) RETURNING id",
            session_id, rol, mensaje,
        )
        return row["id"]


async def get_conversaciones(limit: int = 100) -> list[dict]:
    """Últimas N conversaciones agrupadas por session_id para el panel admin."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, rol, mensaje, timestamp
            FROM conversaciones
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def get_sesiones_recientes(limit: int = 20) -> list[dict]:
    """Últimas sesiones únicas con su último mensaje."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (session_id)
                session_id,
                MAX(timestamp) OVER (PARTITION BY session_id) as ultima_actividad,
                COUNT(*) OVER (PARTITION BY session_id) as total_mensajes
            FROM conversaciones
            ORDER BY session_id, ultima_actividad DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def get_mensajes_sesion(session_id: str) -> list[dict]:
    """Todos los mensajes de una sesión específica."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, rol, mensaje, timestamp FROM conversaciones WHERE session_id=$1 ORDER BY timestamp ASC",
            session_id,
        )
    return [dict(r) for r in rows]


async def existe_conversacion_previa(session_id: str) -> bool:
    """Verifica si ya existe algún log previo para este session_id (teléfono)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM conversaciones WHERE session_id = $1 LIMIT 1)",
            session_id
        )
        return bool(val)


# ── Feedback del doctor ────────────────────────────────────────────

async def guardar_feedback(
    session_id: str,
    respuesta_mala: str,
    correccion: str,
    contexto: str = "",
    nota_generada: str = "",
) -> int:
    """Guarda la corrección del doctor. Devuelve el ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO feedback (session_id, respuesta_mala, contexto, correccion, nota_generada)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
            """,
            session_id, respuesta_mala, contexto, correccion, nota_generada,
        )
        return row["id"]


async def get_feedbacks(limit: int = 50) -> list[dict]:
    """Últimos feedbacks del doctor."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM feedback ORDER BY procesado_en DESC LIMIT $1", limit
        )
    return [dict(r) for r in rows]
