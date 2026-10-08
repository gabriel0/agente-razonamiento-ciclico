# Pre-entrega 5: agente de razonamiento cíclico

Agente de ventas que decide solo cuándo consultar la base de pedidos. Si el
primer corte viene incompleto, vuelve a llamar la herramienta. La sesión queda
en SQLite: con el mismo `thread_id`, la pregunta siguiente reutiliza lo que ya
se habló.

El modelo por defecto es Ollama (`qwen2.5:14b`), el mismo de los módulos
anteriores. No hace falta una API key. Si se quiere un modelo online, la única
opción configurada es Gemini.

## Qué hace el ciclo

La base simulada no entrega el total en la primera lectura. Devuelve la
cantidad de pedidos, `completo=false` y un aviso de información incompleta. El
grafo no tiene un `if` que fuerce el reintento: `tools_condition` mira si el
modelo pidió una herramienta y, si la pidió, ejecuta `buscar_pedidos` y vuelve
al nodo del modelo.

Con el cliente 102 la secuencia es:

1. El usuario pregunta cuántos pedidos tuvo y cuál fue el total.
2. El agente llama `buscar_pedidos(cliente_id=102, incluir_total=false)`.
3. La herramienta responde 3 pedidos, sin total.
4. El agente llama `buscar_pedidos(cliente_id=102, incluir_total=true)`.
5. La herramienta responde 3 pedidos, total 14500 y el último pedido (id 88,
   2 de marzo de 2026, $4.200).
6. El agente responde: Ana Gómez tuvo 3 pedidos por $14.500.

Con el mismo `thread_id` (`sesion-cliente-102`), la pregunta «¿y el último?»
no repite el id. El agente contesta con el pedido 88 y $4.200 a partir del
historial guardado. Esa segunda vuelta no vuelve a llamar la herramienta.

`recursion_limit` está en 10 para que un ciclo mal cerrado no siga indefinido.

## Cómo levantarlo

Hace falta Python 3.12 y Ollama en `http://127.0.0.1:11434`, con el modelo ya
descargado:

```powershell
ollama pull qwen2.5:14b
```

```powershell
cd modulo5\pre-entrega5
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python agente.py
```

La traza queda en `trazas/ejecucion.json`. El checkpoint de la sesión queda en
`data/checkpoints.sqlite` y no se sube al repositorio.

## Gemini

En `.env`:

```
LLM_PROVIDER=google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.5-flash
```

`.env` está en `.gitignore`. `.env.example` no lleva claves.

## Piezas

| Archivo | Rol |
| --- | --- |
| `agente.py` | `AgentState` (hereda de `MessagesState`), herramienta, grafo y prueba |
| `config.py` | Modelo, `thread_id` y `recursion_limit` |
| `trazas/ejecucion.json` | Traza ReAct de la corrida de ejemplo |

El checkpointer es `AsyncSqliteSaver`, la variante asyncio de `SqliteSaver`.
El grafo se compila con ese saver y cada `ainvoke` usa el mismo `thread_id`.
