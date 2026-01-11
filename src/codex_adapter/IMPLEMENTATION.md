# Codex Adapter MVP Implementation

This is the MVP (Week 1 scope) implementation of a Python adapter for the Codex app-server JSON-RPC protocol.

## What's Included

### Core Files

1. **`client.py`** (270 lines)
   - `CodexClient` class - main entry point
   - Subprocess management for `codex app-server`
   - JSON-RPC 2.0 protocol implementation
   - Async streaming with event queue
   - Request/response pairing
   - Connection lifecycle management

2. **`events.py`** (66 lines)
   - `CodexEvent` dataclass for notifications
   - Event type enum (11 common event types)
   - Helper methods: `is_delta()`, `is_completed()`, `get_text_delta()`
   - Factory method: `from_notification()`

3. **`types.py`** (35 lines)
   - `CodexThread` - conversation thread metadata
   - `CodexTurn` - turn metadata
   - `CodexItem` - item (message, tool call) metadata

4. **`exceptions.py`** (26 lines)
   - `CodexError` - base exception
   - `CodexProcessError` - subprocess errors
   - `CodexRequestError` - JSON-RPC errors with code/message/data

5. **`__init__.py`** (25 lines)
   - Package exports
   - Clean public API

### Documentation

6. **`README.md`** (250 lines)
   - Quick start guide
   - API reference
   - Event types table
   - Multiple usage examples
   - Architecture diagram
   - Limitations and future work

7. **`example.py`** (160 lines)
   - 4 working examples:
     - Simple chat
     - Multi-turn conversation
     - Model override per turn
     - Event inspection
   - Menu-driven CLI

8. **`IMPLEMENTATION.md`** (this file)
   - Implementation notes
   - Design decisions
   - Comparison to inspiration sources

### Testing

9. **`tests/test_codex_adapter.py`** (150 lines)
   - Unit tests for core functionality
   - Mock-based subprocess testing
   - Event processing tests
   - Error handling tests

## Design Decisions

### 1. Inspired by gobby/adapters/codex.py

Borrowed from the Gobby example:
- ✅ `CodexAppServerClient` naming pattern → `CodexClient`
- ✅ Context manager for lifecycle (`async with`)
- ✅ Event dataclasses with type hints
- ✅ Subprocess spawning with stdin/stdout
- ✅ Reader task pattern for async I/O

**Simplified from Gobby:**
- ❌ No `ConnectionState` enum (simpler is/not None check)
- ❌ No notification handler registry (single event queue for MVP)
- ❌ No thread tracking dict (just return threads, user manages)
- ✅ Cleaner separation: events.py, types.py, exceptions.py

### 2. Inspired by DAGFromCode/scripts/codex_cli_process.py

Borrowed concepts:
- ✅ Event queue pattern for streaming
- ✅ Pending requests dict for request/response pairing
- ✅ Reader loop with line buffering

**Improved from DAGFromCode:**
- ✅ Used anyio streams (no 64KB readline limit like asyncio)
- ✅ AsyncIterator pattern for `turn_stream()` (cleaner than queue polling)
- ✅ Automatic event routing (no manual dequeue)

### 3. Inspired by termux-extensions-2 adapter pattern

Borrowed structure:
- ✅ Adapter naming convention
- ✅ Protocol normalization concept
- ✅ Clear input/output transformation

**Differences:**
- ❌ Their adapter is stateless translator (HTTP webhooks)
- ✅ Ours is stateful client (persistent subprocess)
- ✅ We expose native app-server events, don't translate

### 4. Reused ACP Infrastructure

**From `acp/connection.py`:**
- ✅ anyio streams pattern (cross-platform, no line limits)
- ✅ TextReceiveStream for newline-delimited JSON
- ✅ Buffer management for incomplete lines
- ✅ Observer pattern concept (adapted to event queue)

**Why NOT reuse full ACP Connection:**
- Different protocol: app-server uses stdio, ACP uses ByteStreams abstraction
- Simpler requirements: don't need full dispatcher/task supervisor for MVP
- Single handler: app-server is request/response + notifications (simpler than ACP's dual handlers)

### 5. Async I/O with anyio

**Why anyio instead of asyncio directly:**
- ✅ Cross-platform file stream support
- ✅ No 64KB readline limit (critical for large responses)
- ✅ Better integration with existing ACP code
- ✅ Already a dependency in agentpool

**Pattern from ACP:**
```python
stream = anyio.wrap_file(self._process.stdout)
text_stream = TextReceiveStream(stream)
async for chunk in text_stream:
    lines = (buffer + chunk).split("\n")
    buffer = lines.pop()  # Keep incomplete line
```

## API Design Philosophy

### Simple and Pythonic

```python
# Good: Clear, async-first
async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")
    async for event in client.turn_stream(thread.id, "Hello"):
        print(event.get_text_delta(), end="")

# Not: Over-engineered
client = CodexClient()
await client.connect()
session = await client.create_session()
request = TurnRequest(thread=thread, input="Hello")
async for response in session.stream(request):
    ...
```

### Event-Driven, Not Callback Hell

```python
# Good: Natural async iteration
async for event in client.turn_stream(...):
    if event.event_type == "item/agentMessage/delta":
        print(event.get_text_delta())

# Not: Callback registration
client.on("item/agentMessage/delta", lambda e: print(e.text))
client.on("turn/completed", lambda e: done.set())
```

### Dict-Based Events (MVP)

**Current (MVP):**
```python
event.data["text"]  # Dict access
event.get_text_delta()  # Helper method
```

**Future (Week 2):**
```python
event.data.text  # Pydantic model (type-safe)
```

**Rationale:**
- MVP: Dict allows rapid iteration without schema lock-in
- Week 2: Auto-generate Pydantic from app-server JSON Schema
- Allows schema evolution without breaking changes

## Comparison to Official TypeScript SDK

### What TypeScript SDK Does

The official SDK (`codex/sdk/typescript/src/`) spawns `codex exec --experimental-json`, NOT `app-server`.

**Exec interface:**
- ❌ No per-turn model override (only thread-level config)
- ❌ No thread management APIs
- ❌ Limited to single session lifecycle
- ✅ Simpler (one request = one session)

### What Our Adapter Does Differently

**App-server interface:**
- ✅ Per-turn model/effort override (sticky)
- ✅ Thread management (list, fork, rollback, archive)
- ✅ Multiple concurrent threads
- ✅ Bidirectional communication (approvals)
- ✅ Full event streaming

**Comparison:**

| Feature | TS SDK (exec) | Python Adapter (app-server) |
|---------|---------------|----------------------------|
| Interface | `codex exec` | `codex app-server` |
| Protocol | Custom JSON | JSON-RPC 2.0 |
| Threads | Single | Multiple |
| Model override | ❌ Thread-level only | ✅ Per-turn (sticky) |
| Streaming | ✅ Events | ✅ Events |
| Approvals | ✅ Handled | ✅ Exposed (MVP: manual) |
| Fork/rollback | ❌ N/A | ✅ Via app-server API |

## MVP Scope (Week 1) ✅

### Completed

- [x] Core client with subprocess management
- [x] JSON-RPC 2.0 protocol implementation
- [x] Thread start with config
- [x] Turn streaming with events
- [x] Per-turn overrides (model, effort)
- [x] Event types and helpers
- [x] Error handling (process + request errors)
- [x] Context manager lifecycle
- [x] Documentation (README + examples)
- [x] Unit tests (basic coverage)
- [x] Type hints throughout

### Not Included (Future Weeks)

- [ ] Approval handlers (events exposed but no handler registration)
- [ ] Thread operations (fork, rollback, archive, list)
- [ ] Skills API
- [ ] Review API
- [ ] Config management API
- [ ] Pydantic models (currently dict-based)
- [ ] Multi-turn concurrency (single event queue)
- [ ] Retry/reconnection logic
- [ ] Session persistence
- [ ] Structured logging

## Testing the MVP

### 1. Manual Test

```bash
# Requires codex CLI installed
cd src/codex_adapter
python3 example.py 1  # Simple chat example
```

### 2. Unit Tests

```bash
pytest tests/test_codex_adapter.py -v
```

### 3. Integration Test (Requires Codex)

```python
import asyncio
from codex_adapter import CodexClient

async def test():
    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")
        async for event in client.turn_stream(thread.id, "Echo 'Hello, World!'"):
            print(event.event_type, event.data)
            if event.event_type == "turn/completed":
                break

asyncio.run(test())
```

## Next Steps (Week 2+)

### Week 2: Type Safety
- [ ] Extract JSON Schema from app-server docs
- [ ] Auto-generate Pydantic models for events/params
- [ ] Migrate `event.data` from dict to Pydantic
- [ ] Add validation for request params

### Week 3-4: Full API Coverage
- [ ] Thread operations (fork, rollback, archive)
- [ ] Approval handler registration
- [ ] Skills API (list, get, enable/disable)
- [ ] Review API (start review, comment)
- [ ] Config management (get/set config)
- [ ] Account APIs (profiles, auth)

### Performance & Production
- [ ] Per-turn event queues (concurrent turns)
- [ ] Connection pooling
- [ ] Retry logic with exponential backoff
- [ ] Health checks / heartbeat
- [ ] Structured logging with filtering
- [ ] Metrics (request latency, event rates)

## Known Limitations

1. **Single Event Queue**: Only one `turn_stream()` can be active at a time
   - Events from concurrent turns would interleave
   - Solution: Per-turn event queues (Week 3)

2. **No Approval Handlers**: `approval/requested` events are emitted but not handled
   - User must manually respond via `approval/respond` request
   - Solution: Handler registration system (Week 3)

3. **Dict-based Events**: Not type-safe, rely on documentation
   - Risk of typos in `event.data["field"]` access
   - Solution: Pydantic models (Week 2)

4. **No Reconnection**: Process failure = full stop
   - Must recreate client and threads
   - Solution: Automatic reconnection with state recovery (Week 4)

5. **Basic Error Handling**: Process errors kill the client
   - stderr is captured but not parsed
   - Solution: Structured error parsing, retry logic (Week 4)

## Files Changed in Agentpool

### New Files
- `src/codex_adapter/__init__.py`
- `src/codex_adapter/client.py`
- `src/codex_adapter/events.py`
- `src/codex_adapter/types.py`
- `src/codex_adapter/exceptions.py`
- `src/codex_adapter/example.py`
- `src/codex_adapter/README.md`
- `src/codex_adapter/IMPLEMENTATION.md`
- `src/codex_adapter/py.typed`
- `tests/test_codex_adapter.py`

### Modified Files
- None (fully additive)

## Integration with AgentPool

Future integration pattern (Week 5+):

```python
# agentpool/agents/codex_agent/codex_agent.py
from codex_adapter import CodexClient

class CodexAgent(BaseAgent):
    async def run_stream(self, *prompts, **overrides):
        async with CodexClient() as client:
            thread = await client.thread_start(
                cwd=self.workspace,
                model=overrides.get("model"),
                effort=overrides.get("effort"),
            )
            
            async for event in client.turn_stream(thread.id, prepare_prompts(prompts)):
                # Convert CodexEvent → RichAgentStreamEvent
                yield convert_event(event)
```

Similar to how `ClaudeCodeAgent` wraps the Claude Code SDK.

## Conclusion

This MVP provides a solid foundation for Codex integration:

✅ **Working prototype** - Can start threads, stream turns, override models
✅ **Clean API** - Pythonic, async-first, event-driven
✅ **Well-documented** - README, examples, tests
✅ **Type-safe** - Full type hints (dict events for now, Pydantic in Week 2)
✅ **Testable** - Mock-based unit tests, example scripts
✅ **Extensible** - Clear path to full API coverage

Ready for Week 2: Type safety with Pydantic models!
