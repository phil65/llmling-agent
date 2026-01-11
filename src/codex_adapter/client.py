"""Codex app-server client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator  # noqa: TC003
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter

from codex_adapter.codex_types import CodexThread
from codex_adapter.events import CodexEvent
from codex_adapter.exceptions import CodexProcessError, CodexRequestError
from codex_adapter.models import (
    ClientInfo,
    CommandExecParams,
    CommandExecResponse,
    InitializeParams,
    JsonRpcRequest,
    JsonRpcResponse,
    ModelData,
    ModelListResponse,
    SkillData,
    SkillsListParams,
    SkillsListResponse,
    TextInputItem,
    ThreadArchiveParams,
    ThreadForkParams,
    ThreadListParams,
    ThreadListResponse,
    ThreadLoadedListResponse,
    ThreadResponse,
    ThreadResumeParams,
    ThreadRollbackParams,
    ThreadRollbackResponse,
    ThreadStartParams,
    TurnInputItem,
    TurnInterruptParams,
    TurnStartParams,
    TurnStartResponse,
)


# Type aliases for API parameters
ReasoningEffort = Literal["low", "medium", "high"]
ApprovalPolicy = Literal["always", "never", "auto"]

ResultType = TypeVar("ResultType", bound=BaseModel)

if TYPE_CHECKING:
    from typing import Self

logger = logging.getLogger(__name__)


class CodexClient:
    """Client for the Codex app-server JSON-RPC protocol.

    Manages the subprocess lifecycle and provides async methods for:
    - Thread management (conversations)
    - Turn management (message exchanges)
    - Event streaming via notifications

    Example:
        async with CodexClient() as client:
            thread = await client.thread_start(cwd="/path/to/project")
            async for event in client.turn_stream(thread.id, "Help me refactor"):
                print(event.get_text_delta(), end="", flush=True)
    """

    def __init__(
        self,
        codex_command: str = "codex",
        profile: str | None = None,
    ) -> None:
        """Initialize the Codex app-server client.

        Args:
            codex_command: Path to the codex binary (default: "codex")
            profile: Optional Codex profile to use
        """
        self._codex_command = codex_command
        self._profile = profile
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}
        self._event_queue: asyncio.Queue[CodexEvent | None] = asyncio.Queue()
        self._turn_queues: dict[str, asyncio.Queue[CodexEvent | None]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_lock = asyncio.Lock()
        self._active_threads: dict[str, CodexThread] = {}
        self._global_event_handlers: list[Any] = []  # For global events

    async def __aenter__(self) -> Self:
        """Async context manager entry - starts the app-server."""
        await self.start()
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Async context manager exit - stops the app-server."""
        await self.stop()

    async def start(self) -> None:
        """Start the Codex app-server subprocess and initialize connection.

        Raises:
            CodexProcessError: If failed to start the process
        """
        if self._process is not None:
            return

        cmd = [self._codex_command, "app-server"]
        if self._profile:
            cmd.extend(["--profile", self._profile])

        logger.info("Starting Codex app-server: %s", " ".join(cmd))

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise CodexProcessError(f"Codex binary not found: {self._codex_command}") from exc
        except Exception as exc:
            raise CodexProcessError(f"Failed to start Codex app-server: {exc}") from exc

        # Start reader task
        self._reader_task = asyncio.create_task(self._read_loop())

        # Initialize connection
        init_params = InitializeParams(
            client_info=ClientInfo(
                name="agentpool-codex-adapter",
                version="0.1.0",
            )
        )
        await self._send_request("initialize", init_params)

    async def stop(self) -> None:
        """Stop the Codex app-server subprocess."""
        if self._process is None:
            return

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        # Terminate process
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

        self._process = None

        # Reject pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(CodexProcessError("Connection closed"))
        self._pending_requests.clear()

    async def thread_start(
        self,
        *,
        cwd: str | None = None,
        model: str | None = None,
        effort: ReasoningEffort | None = None,
    ) -> CodexThread:
        """Start a new conversation thread.

        Args:
            cwd: Working directory for the thread
            model: Model to use (e.g., "gpt-5-codex")
            effort: Reasoning effort

        Returns:
            CodexThread: The created thread
        """
        params = ThreadStartParams(cwd=cwd, model=model, effort=effort)
        result = await self._send_request("thread/start", params)
        response = ThreadResponse.model_validate(result)
        thread = CodexThread(
            id=response.thread.id,
            preview=response.thread.preview,
            model_provider=response.thread.model_provider,
            created_at=response.thread.created_at,
        )
        self._active_threads[thread.id] = thread
        return thread

    async def thread_resume(self, thread_id: str) -> CodexThread:
        """Resume an existing thread by ID.

        Args:
            thread_id: The thread ID to resume

        Returns:
            CodexThread: The resumed thread
        """
        params = ThreadResumeParams(thread_id=thread_id)
        result = await self._send_request("thread/resume", params)
        response = ThreadResponse.model_validate(result)
        thread = CodexThread(
            id=response.thread.id,
            preview=response.thread.preview,
            model_provider=response.thread.model_provider,
            created_at=response.thread.created_at,
        )
        self._active_threads[thread.id] = thread
        return thread

    async def thread_fork(self, thread_id: str) -> CodexThread:
        """Fork an existing thread into a new thread with copied history.

        Args:
            thread_id: The thread ID to fork from

        Returns:
            CodexThread: The new forked thread
        """
        params = ThreadForkParams(thread_id=thread_id)
        result = await self._send_request("thread/fork", params)
        response = ThreadResponse.model_validate(result)
        thread = CodexThread(
            id=response.thread.id,
            preview=response.thread.preview,
            model_provider=response.thread.model_provider,
            created_at=response.thread.created_at,
        )
        self._active_threads[thread.id] = thread
        return thread

    async def thread_list(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        model_providers: list[str] | None = None,
    ) -> ThreadListResponse:
        """List stored threads with pagination.

        Args:
            cursor: Opaque pagination cursor from previous response
            limit: Maximum number of threads to return
            model_providers: Filter by model providers (e.g., ["openai", "anthropic"])

        Returns:
            ThreadListResponse with data (list of threads) and next_cursor
        """
        params = ThreadListParams(
            cursor=cursor,
            limit=limit,
            model_providers=model_providers,
        )
        result = await self._send_request("thread/list", params)
        return ThreadListResponse.model_validate(result)

    async def thread_loaded_list(self) -> list[str]:
        """List thread IDs currently loaded in memory.

        Returns:
            List of thread IDs
        """
        result = await self._send_request("thread/loaded/list")
        response = ThreadLoadedListResponse.model_validate(result)
        return response.data

    async def thread_archive(self, thread_id: str) -> None:
        """Archive a thread (move to archived directory).

        Args:
            thread_id: The thread ID to archive
        """
        params = ThreadArchiveParams(thread_id=thread_id)
        await self._send_request("thread/archive", params)
        if thread_id in self._active_threads:
            del self._active_threads[thread_id]

    async def thread_rollback(self, thread_id: str, turns: int) -> ThreadRollbackResponse:
        """Rollback the last N turns from a thread.

        Args:
            thread_id: The thread ID
            turns: Number of turns to rollback

        Returns:
            Updated thread object with turns populated
        """
        params = ThreadRollbackParams(thread_id=thread_id, turns=turns)
        result = await self._send_request("thread/rollback", params)
        return ThreadRollbackResponse.model_validate(result)

    async def turn_stream(
        self,
        thread_id: str,
        user_input: str | list[TurnInputItem],
        *,
        model: str | None = None,
        effort: ReasoningEffort | None = None,
        approval_policy: ApprovalPolicy | None = None,
        output_schema: dict[str, Any] | type[Any] | None = None,
    ) -> AsyncIterator[CodexEvent]:
        """Start a turn and stream events.

        Args:
            thread_id: The thread ID to send the turn to
            user_input: User input as string or list of input items (text/image)
            model: Optional model override for this turn
            effort: Optional reasoning effort override
            approval_policy: Optional approval policy
            output_schema: Optional JSON Schema dict or Pydantic type to constrain output

        Yields:
            CodexEvent: Streaming events from the turn
        """
        # Convert user_input to typed input format
        input_items: list[TurnInputItem]
        if isinstance(user_input, str):
            input_items = [TextInputItem(text=user_input)]
        else:
            input_items = user_input

        # Handle output_schema - convert type to JSON Schema if needed
        schema_dict: dict[str, Any] | None = None
        if output_schema is not None:
            if isinstance(output_schema, dict):
                schema_dict = output_schema
            else:
                # It's a type - use TypeAdapter to extract schema
                adapter = TypeAdapter(output_schema)
                schema_dict = adapter.json_schema()

        # Build typed params
        params = TurnStartParams(
            thread_id=thread_id,
            input=input_items,
            model=model,
            effort=effort,
            approval_policy=approval_policy,
            output_schema=schema_dict,
        )

        # Start turn (non-blocking request)
        turn_result = await self._send_request("turn/start", params)
        response = TurnStartResponse.model_validate(turn_result)
        turn_id = response.turn.id

        # Create per-turn event queue for proper routing
        turn_queue: asyncio.Queue[CodexEvent | None] = asyncio.Queue()
        turn_key = f"{thread_id}:{turn_id}"
        self._turn_queues[turn_key] = turn_queue

        try:
            # Stream events until turn completes
            while True:
                event = await turn_queue.get()
                if event is None:
                    break

                yield event

                # Check for turn completion
                if event.event_type == "turn/completed":
                    break
                elif event.event_type == "turn/error":
                    error_msg = event.data.get("error", "Unknown error")
                    raise CodexRequestError(-32000, error_msg)
        finally:
            # Cleanup turn queue
            if turn_key in self._turn_queues:
                del self._turn_queues[turn_key]

    async def turn_interrupt(self, thread_id: str, turn_id: str) -> None:
        """Interrupt a running turn.

        Args:
            thread_id: The thread ID
            turn_id: The turn ID to interrupt
        """
        params = TurnInterruptParams(thread_id=thread_id, turn_id=turn_id)
        await self._send_request("turn/interrupt", params)

    async def turn_stream_structured(
        self,
        thread_id: str,
        user_input: str | list[TurnInputItem],
        result_type: type[ResultType],
        *,
        model: str | None = None,
        effort: ReasoningEffort | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> ResultType:
        """Start a turn with structured output and return the parsed result.

        This is a convenience method that combines turn_stream with automatic
        schema generation and result parsing. Similar to PydanticAI's approach.

        Note: This method only accepts Pydantic types (not raw dict schemas).
        For dict schemas, use turn_stream() with output_schema and parse manually.

        Args:
            thread_id: The thread ID to send the turn to
            user_input: User input as string or list of items
            result_type: Pydantic model class for the expected result (not a dict!)
            model: Optional model override for this turn
            effort: Optional reasoning effort override
            approval_policy: Optional approval policy

        Returns:
            Parsed Pydantic model instance of type result_type

        Raises:
            ValidationError: If the agent's response doesn't match the schema
            CodexRequestError: If the turn fails

        Example:
            class FileInfo(BaseModel):
                name: str
                type: str

            class FileList(BaseModel):
                files: list[FileInfo]
                total: int

            result = await client.turn_stream_structured(
                thread.id,
                "List Python files in current directory",
                FileList,  # Must be a Pydantic type, not a dict
            )
            print(f"Found {result.total} files: {result.files}")
        """
        # Collect agent message text
        response_text = ""
        async for event in self.turn_stream(
            thread_id,
            user_input,
            model=model,
            effort=effort,
            approval_policy=approval_policy,
            output_schema=result_type,  # Auto-generate schema from type
        ):
            if event.event_type == "item/agentMessage/delta":
                response_text += event.get_text_delta()
            elif event.event_type == "turn/error":
                error_msg = event.data.get("error", "Unknown error")
                raise CodexRequestError(-32000, error_msg)

        # Parse into typed model
        return result_type.model_validate_json(response_text)

    async def skills_list(
        self,
        *,
        cwd: str | None = None,
        force_reload: bool = False,
    ) -> list[SkillData]:
        """List available skills.

        Args:
            cwd: Optional working directory to scope skills
            force_reload: Force reload of skills cache

        Returns:
            List of skills with name and description
        """
        params = SkillsListParams(cwd=cwd, force_reload=force_reload)
        result = await self._send_request("skills/list", params)
        response = SkillsListResponse.model_validate(result)
        # Return skills from first container (usually only one)
        if response.data:
            return response.data[0].skills
        return []

    async def model_list(self) -> list[ModelData]:
        """List available models with reasoning effort options.

        Returns:
            List of available models
        """
        result = await self._send_request("model/list")
        response = ModelListResponse.model_validate(result)
        return response.data

    async def command_exec(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        sandbox_policy: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> CommandExecResponse:
        """Execute a command without creating a thread/turn.

        Args:
            command: Command and arguments as list (e.g., ["ls", "-la"])
            cwd: Working directory for command
            sandbox_policy: Sandbox policy override
            timeout_ms: Timeout in milliseconds

        Returns:
            CommandExecResponse with exit_code, stdout, stderr
        """
        params = CommandExecParams(
            command=command,
            cwd=cwd,
            sandbox_policy=sandbox_policy,
            timeout_ms=timeout_ms,
        )
        result = await self._send_request("command/exec", params)
        return CommandExecResponse.model_validate(result)

    async def _send_request(
        self,
        method: str,
        params: BaseModel | None = None,
    ) -> Any:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: JSON-RPC method name
            params: Pydantic model with request parameters (will be serialized)

        Returns:
            Response result (not yet validated - caller should validate)
        """
        if self._process is None or self._process.stdin is None:
            raise CodexProcessError("Not connected to Codex app-server")

        request_id = self._request_id
        self._request_id += 1

        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[request_id] = future

        # Serialize params to dict if provided
        params_dict: dict[str, Any] = {}
        if params is not None:
            params_dict = params.model_dump(by_alias=True, exclude_none=True)

        # Build JSON-RPC request
        request = JsonRpcRequest(
            id=request_id,
            method=method,
            params=params_dict,
        )

        async with self._writer_lock:
            try:
                line = request.model_dump_json(by_alias=True, exclude_none=True) + "\n"
                self._process.stdin.write(line.encode())
                await self._process.stdin.drain()
            except Exception as exc:
                del self._pending_requests[request_id]
                raise CodexProcessError(f"Failed to send request: {exc}") from exc

        return await future

    async def _read_loop(self) -> None:
        """Read messages from app-server stdout."""
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").strip()
                if not line or line == "null":
                    continue

                try:
                    message = json.loads(line)
                    await self._process_message(message)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON: %s", line)
                except Exception:
                    logger.exception("Error processing message")

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Reader loop failed")
        finally:
            await self._event_queue.put(None)

    async def _process_message(self, message: dict[str, Any]) -> None:
        """Process a message from the app-server.

        Args:
            message: Raw JSON-RPC message (response or notification)
        """
        # Response (has "id" field)
        if "id" in message:
            try:
                response = JsonRpcResponse.model_validate(message)
                request_id = response.id
                future = self._pending_requests.pop(request_id, None)

                if future and not future.done():
                    if response.error:
                        future.set_exception(
                            CodexRequestError(
                                response.error.code,
                                response.error.message,
                                response.error.data,
                            )
                        )
                    else:
                        future.set_result(response.result)
            except Exception as exc:
                logger.warning("Failed to parse response: %s", exc)
                # Fallback to old behavior for unrecognized responses
                fallback_id = message.get("id")
                if fallback_id is not None and isinstance(fallback_id, int):
                    future = self._pending_requests.pop(fallback_id, None)
                    if future and not future.done():
                        future.set_result(message.get("result"))
            return

        # Notification (has "method" field, no "id")
        if "method" in message:
            method = message["method"]
            params = message.get("params") or {}
            event = CodexEvent.from_notification(method, params)

            # Route event to appropriate turn queue
            thread_id = params.get("threadId")
            turn_id = params.get("turnId")

            # Also check nested turn object (some events have it there)
            if not turn_id and "turn" in params:
                turn_data = params.get("turn", {})
                if isinstance(turn_data, dict):
                    turn_id = turn_data.get("id")

            if thread_id and turn_id:
                # Turn-specific event - route to turn queue
                turn_key = f"{thread_id}:{turn_id}"
                if turn_key in self._turn_queues:
                    await self._turn_queues[turn_key].put(event)
                else:
                    # Turn queue not found (might be old event) - put in global queue
                    await self._event_queue.put(event)
            else:
                # Global event (account, MCP, etc.) - put in global queue
                await self._event_queue.put(event)
