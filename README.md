# Mateo — agente de WhatsApp para un consultorio médico

Agente de IA que atiende el WhatsApp de un consultorio: contesta dudas (precios, horarios, preparación para cirugía), agenda citas revisando disponibilidad real y manda recordatorios antes de la cita. Lo hice durante mis prácticas en una empresa de desarrollo y estuvo funcionando con un consultorio real.

> Los datos del consultorio en este repo son ficticios (Dr. Alejandro Martínez). Cambié el nombre del cliente real.

## Cómo funciona

```
Paciente (WhatsApp) ──► Kapso (webhook) ──► FastAPI ──► Claude
                                              │
                         ┌────────────────────┼─────────────────────┐
                         ▼                    ▼                     ▼
                  vault/*.md (Obsidian)   MongoDB Atlas        PostgreSQL
                  identidad, precios,     citas y              historial de
                  horarios, políticas     disponibilidad       conversaciones
```

- **Base de conocimiento en Obsidian.** Toda la info del consultorio vive en notas `.md` (`vault/`). Si el doctor cambia un precio, edita la nota y el bot ya responde con el dato nuevo, sin tocar código.
- **Agendado con validación.** Antes de confirmar una cita revisa que el horario esté dentro del horario del consultorio, que no esté en el pasado y que no choque con otra cita en Mongo. La cita se guarda hasta que el modelo confirma, no antes.
- **Recordatorios.** Un loop en segundo plano busca citas próximas y le manda recordatorio al paciente por WhatsApp.
- **Recurrencia.** Con el historial en Postgres sabe si el paciente ya había escrito, para no presentarse otra vez.
- **Panel admin** (`/admin/citas`, `/admin/conversaciones`, `/admin/logs`) para que el doctor vea citas y conversaciones. El doctor también puede dejar correcciones que se guardan como notas en el vault.
- **API de conocimiento** (`/api/conocimiento`): sirve el vault de cualquier negocio (médico, estética, barbería) para que otros agentes lo reutilicen.

## Stack

Python · FastAPI · Claude API (Anthropic) · Kapso (WhatsApp Business API) · MongoDB Atlas (motor) · PostgreSQL (asyncpg) · Obsidian · Render

## Archivos

| Archivo | Qué hace |
|---|---|
| `agente_kapso.py` | Servidor de producción: webhook de WhatsApp, prompt, agendado, recordatorios, endpoints admin |
| `db_mongo.py` | Citas: guardar, buscar, detectar conflictos |
| `db_postgres.py` | Logs de conversación y feedback |
| `admin.py` / `admin.html` | Panel para revisar sesiones y dejar correcciones |
| `agente.py` + `indexar.py` | Primera versión local con ChromaDB + embeddings (RAG vectorial) |
| `vault/` | La "memoria" del consultorio en Markdown |

## Cosas que aprendí

- La primera versión usaba ChromaDB con `sentence-transformers` para RAG vectorial. En Render (plan de 512 MB) no cabía con PyTorch, así que cambié a leer el vault directo. Para una base de conocimiento chica funciona igual de bien y es mucho más ligero.
- Antes de usarlo con un consultorio real lo validé con un prototipo para una barbería.
- Gran parte del trabajo fue afinar el prompt: mensajes cortos que suenen humanos, nunca dar diagnósticos y no inventar horarios.

## Correrlo

```bash
pip install -r requirements.txt
# variables: ANTHROPIC_API_KEY, KAPSO_API_KEY, KAPSO_PHONE_NUMBER_ID, MONGODB_URI, POSTGRES_URL, ADMIN_PASSWORD
uvicorn agente_kapso:app --port 8002
```

Pendiente: reactivar la verificación de firma del webhook de Kapso (la desactivé mientras depuraba un 401).
