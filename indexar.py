from pathlib import Path
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

console = Console()

VAULT_PATH  = Path(r"C:\Users\gaelg_12gxbrs\OneDrive\Documentos\Doctor-Agente")
CHROMA_PATH = Path("./chroma_db")


def indexar(force: bool = False):
    if CHROMA_PATH.exists():
        if not force:
            console.print("[yellow]Ya existe un índice en ./chroma_db[/yellow]")
            respuesta = input("¿Deseas re-indexar? (s/n): ").strip().lower()
            if respuesta != "s":
                console.print("[yellow]Indexación cancelada.[/yellow]")
                return
        shutil.rmtree(CHROMA_PATH)
        console.print("[cyan]Índice anterior eliminado.[/cyan]")

    md_files = list(VAULT_PATH.rglob("*.md"))
    if not md_files:
        console.print(f"[red]No se encontraron archivos .md en {VAULT_PATH}[/red]")
        return

    console.print(f"[green]Encontrados {len(md_files)} archivos .md[/green]")
    console.print("[cyan]Cargando modelo de embeddings (all-MiniLM-L6-v2)...[/cyan]")

    model         = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection    = chroma_client.create_collection("doctor")

    for md_file in track(md_files, description="Indexando notas..."):
        content   = md_file.read_text(encoding="utf-8")
        embedding = model.encode(content).tolist()
        collection.add(
            documents=[content],
            embeddings=[embedding],
            ids=[md_file.stem],
            metadatas=[{"nombre": md_file.stem, "archivo": md_file.name}],
        )
        console.print(f"  [dim]OK {md_file.name}[/dim]")

    console.print(f"[bold green]Indexación completada — {len(md_files)} notas guardadas en ChromaDB.[/bold green]")


def indexar_nota_individual(ruta_archivo: str):
    """Indexa o actualiza una sola nota en ChromaDB sin tocar las demás."""
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        console.print(f"[red]Archivo no encontrado: {ruta}[/red]")
        return

    model         = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = chroma_client.get_collection("doctor")
    except Exception:
        collection = chroma_client.create_collection("doctor")

    content   = ruta.read_text(encoding="utf-8")
    embedding = model.encode(content).tolist()
    nota_id   = ruta.stem

    # Si ya existe la actualiza, si no la agrega
    try:
        collection.update(
            documents=[content],
            embeddings=[embedding],
            ids=[nota_id],
            metadatas=[{"nombre": nota_id, "archivo": ruta.name}],
        )
        console.print(f"[cyan]Nota actualizada en ChromaDB: {ruta.name}[/cyan]")
    except Exception:
        collection.add(
            documents=[content],
            embeddings=[embedding],
            ids=[nota_id],
            metadatas=[{"nombre": nota_id, "archivo": ruta.name}],
        )
        console.print(f"[green]Nota nueva indexada en ChromaDB: {ruta.name}[/green]")


if __name__ == "__main__":
    import sys
    indexar(force="--force" in sys.argv)
