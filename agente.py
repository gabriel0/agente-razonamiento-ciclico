"""Agente ReAct de ventas: busca pedidos, reintenta si el corte viene incompleto
y recuerda la sesión con SqliteSaver."""

import asyncio
import json
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from config import (
    DB_PATH,
    RECURSION_LIMIT,
    THREAD_ID,
    TRAZA_PATH,
    build_chat_model,
    nombre_modelo,
)

SYSTEM = SystemMessage(
    content=(
        "Sos un analista de ventas. No inventes cifras ni clientes.\n"
        "Para pedidos y totales usá buscar_pedidos.\n"
        "La primera respuesta de esa herramienta es un corte parcial: "
        "completo=false y no trae el total. Cuando veas informacion_incompleta "
        "o completo=false, llamá otra vez a buscar_pedidos con el mismo "
        "cliente_id e incluir_total=true. No respondas al usuario antes de eso.\n"
        "Si la herramienta dice que el cliente no existe, pedí que aclaren el id.\n"
        "Si después preguntan por el último pedido, usá el cliente y los datos "
        "que ya están en la conversación. No vuelvas a pedir el id.\n"
        "Respondé en español y formateá los miles con punto: $14.500."
    )
)


class Pedido(TypedDict):
    id: int
    fecha: str
    monto: int


class FichaCliente(TypedDict):
    cliente: str
    pedidos: list[Pedido]


PEDIDOS: dict[int, FichaCliente] = {
    102: {
        "cliente": "Ana Gómez",
        "pedidos": [
            {"id": 86, "fecha": "2026-01-14", "monto": 5300},
            {"id": 87, "fecha": "2026-02-03", "monto": 5000},
            {"id": 88, "fecha": "2026-03-02", "monto": 4200},
        ],
    },
    215: {
        "cliente": "Luis Pérez",
        "pedidos": [
            {"id": 91, "fecha": "2026-02-20", "monto": 1800},
        ],
    },
}

# La base simulada entrega el total recién en el segundo intento.
_cortes_parciales: set[int] = set()


class BuscarPedidosInput(BaseModel):
    """Argumentos de la búsqueda de pedidos."""

    cliente_id: int = Field(description="Id numérico del cliente. Ejemplo: 102.")
    incluir_total: bool = Field(
        default=False,
        description=(
            "false en la primera llamada. true solo en el reintento, "
            "después de recibir informacion_incompleta."
        ),
    )


class AgentState(MessagesState):
    """Historial de la sesión. messages se acumula con add_messages."""


@tool(args_schema=BuscarPedidosInput)
async def buscar_pedidos(cliente_id: int, incluir_total: bool = False) -> str:
    """Busca en la base de ventas cuántos pedidos tuvo un cliente y, en un segundo intento, el total.

    Usala cuando pregunten por la cantidad de pedidos, el monto total o el último pedido.
    El cliente se identifica solo por cliente_id (un entero, por ejemplo 102). No acepta nombres.

    La primera llamada para un cliente siempre devuelve un corte parcial: la cantidad de
    pedidos, completo=false y el campo informacion_incompleta. En esa respuesta el total
    no está. No alcances una conclusión con ese corte.

    Si completo es false, llamá otra vez a esta misma herramienta con el mismo cliente_id
    e incluir_total=true. Esa segunda llamada trae pedidos, total y ultimo.

    Si el cliente no existe, la herramienta devuelve error. En ese caso pedile al usuario
    que aclare el id. No pruebes otros ids por tu cuenta.
    """
    ficha = PEDIDOS.get(cliente_id)
    if ficha is None:
        return json.dumps(
            {
                "error": f"No existe el cliente {cliente_id}.",
                "completo": False,
                "sugerencia": "Pedile al usuario que aclare el id. No inventes otro.",
            },
            ensure_ascii=False,
        )

    if cliente_id not in _cortes_parciales or not incluir_total:
        _cortes_parciales.add(cliente_id)
        return json.dumps(
            {
                "cliente_id": cliente_id,
                "pedidos": len(ficha["pedidos"]),
                "total": None,
                "ultimo": None,
                "completo": False,
                "informacion_incompleta": (
                    "Corte parcial de la base: falta el total. No respondas todavía. "
                    f"Llamá de nuevo a buscar_pedidos con cliente_id={cliente_id} "
                    "e incluir_total=true."
                ),
            },
            ensure_ascii=False,
        )

    pedidos = ficha["pedidos"]
    return json.dumps(
        {
            "cliente_id": cliente_id,
            "cliente": ficha["cliente"],
            "pedidos": len(pedidos),
            "total": sum(pedido["monto"] for pedido in pedidos),
            "ultimo": pedidos[-1],
            "completo": True,
        },
        ensure_ascii=False,
    )


tools = [buscar_pedidos]
model = build_chat_model().bind_tools(tools)


async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
    respuesta = await model.ainvoke(state["messages"])
    return {"messages": [respuesta]}


def build_workflow() -> StateGraph[AgentState]:
    """Modelo y herramientas unidos por tools_condition. El ciclo vuelve de tools al modelo."""
    workflow: StateGraph[AgentState] = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow


def serializar_mensaje(mensaje: BaseMessage) -> dict[str, Any]:
    dato: dict[str, Any] = {"type": mensaje.type, "content": mensaje.content}
    llamadas = getattr(mensaje, "tool_calls", None)
    if llamadas:
        dato["tool_calls"] = llamadas
    nombre = getattr(mensaje, "name", None)
    if nombre:
        dato["name"] = nombre
    return dato


def mostrar(titulo: str, mensajes: list[BaseMessage]) -> None:
    print(titulo)
    for mensaje in mensajes:
        llamadas = getattr(mensaje, "tool_calls", None) or []
        detalle = ""
        if llamadas:
            detalle = " " + ", ".join(
                f"{llamada['name']}({llamada['args']})" for llamada in llamadas
            )
        print(f"[{mensaje.type}] {mensaje.content}{detalle}")


def contar_tools(mensajes: list[BaseMessage]) -> int:
    return sum(1 for mensaje in mensajes if mensaje.type == "tool")


def limpiar_sqlite() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    for sufijo in ("", "-journal", "-wal", "-shm"):
        archivo = DB_PATH.parent / f"{DB_PATH.name}{sufijo}"
        if archivo.exists():
            archivo.unlink()


async def main() -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    limpiar_sqlite()
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
    }
    preguntas = [
        "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?",
        "¿y el último?",
    ]
    traza: dict[str, Any] = {
        "thread_id": THREAD_ID,
        "recursion_limit": RECURSION_LIMIT,
        "modelo": nombre_modelo(),
        "checkpointer": "AsyncSqliteSaver",
        "turnos": [],
    }

    # AsyncSqliteSaver es la variante asyncio de SqliteSaver.
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memoria:
        app = build_workflow().compile(checkpointer=memoria)
        historial: list[BaseMessage] = []

        for numero, pregunta in enumerate(preguntas, start=1):
            mensajes: list[BaseMessage] = [HumanMessage(content=pregunta)]
            if numero == 1:
                mensajes.insert(0, SYSTEM)

            print(f"--- Paso {numero} ---")
            resultado = await app.ainvoke({"messages": mensajes}, config=config)
            historial = list(resultado["messages"])
            mostrar(f"Traza del paso {numero}:", historial)

            anteriores = sum(turno["herramientas"] for turno in traza["turnos"])
            herramientas = contar_tools(historial) - anteriores
            traza["turnos"].append(
                {
                    "turno": numero,
                    "pregunta": pregunta,
                    "herramientas": herramientas,
                    "mensajes": [serializar_mensaje(mensaje) for mensaje in historial],
                }
            )

    TRAZA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAZA_PATH.write_text(
        json.dumps(traza, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Traza guardada en {TRAZA_PATH}")

    if traza["turnos"][0]["herramientas"] < 2:
        raise SystemExit(
            "El primer turno tiene que llamar a buscar_pedidos al menos dos veces."
        )


if __name__ == "__main__":
    asyncio.run(main())
