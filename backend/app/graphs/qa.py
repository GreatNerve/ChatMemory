from app.core.gpu_lock import GpuBusyError, gpu_lock

from app.core.schemas import AskResponse

from app.services import rag_chain




async def run_qa(

    workspace_id: str,

    question: str,

    *,

    speaker: str | None = None,

    date_from: str | None = None,

    date_to: str | None = None,

) -> AskResponse:

    try:

        async with gpu_lock(f"qa:{workspace_id}"):

            return rag_chain.run_qa_pipeline(

                workspace_id,

                question,

                speaker=speaker,

                date_from=date_from,

                date_to=date_to,

            )

    except GpuBusyError as exc:

        raise exc

