# Question Tool Bug Analysis and Fix

## Problem

The question tool is not working because of a mismatch between what the tool sends and what the input provider accepts.

### Current Flow

1. **Question tool** (`src/agentpool/tool_impls/question/tool.py`):
   - Creates elicitation with schema: `{"type": "string"}` when no response_schema is provided
   - Calls `ctx.handle_elicitation(params)`

2. **AgentContext** (`src/agentpool/agents/context.py:96`):
   - Forwards to input provider: `provider.get_elicitation(params)`

3. **ACPInputProvider** (`src/agentpool_server/opencode_server/input_provider.py:215`):
   - Only handles schemas with `enum` field
   - Returns `ElicitResult(action="decline")` for plain string schemas
   - Never broadcasts question event to client

### Why It Fails

```python
# In input_provider.py get_elicitation():
if isinstance(params, types.ElicitRequestFormParams):
    schema = params.requestedSchema
    
    # Check if schema defines options (enum)
    enum_values = schema.get("enum")
    if enum_values:
        return await self._handle_question_elicitation(params, schema)
    
    # ... more enum checks ...

# For other form elicitation, we don't have UI support yet
return types.ElicitResult(action="decline")  # <-- THIS IS WHERE IT FAILS
```

The tool sends `{"type": "string"}` but the provider only accepts schemas with enum/options.

## Solutions

### Option 1: Support Free-Form Text Input (Recommended)

Modify `ACPInputProvider.get_elicitation()` to handle plain text prompts without enum:

```python
async def get_elicitation(
    self,
    params: types.ElicitRequestParams,
) -> types.ElicitResult | types.ErrorData:
    """Get user response to elicitation request via OpenCode questions."""
    
    # For URL elicitation
    if isinstance(params, types.ElicitRequestURLParams):
        # ... existing code ...
        return types.ElicitResult(action="decline")

    # For form elicitation
    if isinstance(params, types.ElicitRequestFormParams):
        schema = params.requestedSchema

        # Check if schema defines options (enum)
        enum_values = schema.get("enum")
        if enum_values:
            return await self._handle_question_elicitation(params, schema)

        # Check if it's an array schema with enum items
        if schema.get("type") == "array":
            items = schema.get("items", {})
            if items.get("enum"):
                return await self._handle_question_elicitation(params, schema)
        
        # NEW: Handle free-form text input
        if schema.get("type") == "string":
            return await self._handle_text_input_elicitation(params)

    return types.ElicitResult(action="decline")
```

Then add a new method:

```python
async def _handle_text_input_elicitation(
    self,
    params: types.ElicitRequestFormParams,
) -> types.ElicitResult | types.ErrorData:
    """Handle free-form text input via OpenCode input system.
    
    For prompts without predefined options, we can either:
    1. Use a simple text input (if OpenCode supports it)
    2. Create a single "Other" option that accepts free text
    """
    import asyncio
    from agentpool_server.opencode_server.models.events import QuestionAskedEvent
    from agentpool_server.opencode_server.models.question import (
        QuestionInfo,
        QuestionOption,
    )

    question_id = self._generate_permission_id()
    
    # Create a question with a single "Other (type your answer)" option
    question_info = QuestionInfo(
        question=params.message,
        header=params.message[:12],
        options=[
            QuestionOption(
                label="Other",
                description="Type your answer",
            )
        ],
        multiple=None,  # Single answer expected
    )

    # Create future to wait for answer
    future: asyncio.Future[list[list[str]]] = asyncio.get_event_loop().create_future()

    # Store pending question
    from agentpool_server.opencode_server.state import PendingQuestion
    self.state.pending_questions[question_id] = PendingQuestion(
        session_id=self.session_id,
        questions=[question_info],
        future=future,
        tool=None,
    )

    # Broadcast event
    event = QuestionAskedEvent.create(
        request_id=question_id,
        session_id=self.session_id,
        questions=[question_info.model_dump(mode="json", by_alias=True)],
    )
    await self.state.broadcast_event(event)

    logger.info("Text input question asked", question_id=question_id, message=params.message)

    # Wait for answer
    try:
        answers = await future
        answer = answers[0][0] if answers and answers[0] else ""
        
        # Return the free-form text
        content: dict[str, str] = {"value": answer}
        return types.ElicitResult(action="accept", content=content)
    except asyncio.CancelledError:
        logger.info("Question cancelled", question_id=question_id)
        return types.ElicitResult(action="cancel")
    except Exception as e:
        logger.exception("Question failed", question_id=question_id)
        return types.ErrorData(code=-1, message=f"Elicitation failed: {e}")
    finally:
        # Clean up pending question
        self.state.pending_questions.pop(question_id, None)
```

### Option 2: Use response_schema Parameter

Update the question tool to always provide an enum with an "Other" option:

```python
async def _execute(
    self,
    ctx: AgentContext,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> ToolResult:
    """Ask the user a clarifying question."""
    from mcp.types import ElicitRequestFormParams, ElicitResult, ErrorData

    # If no schema provided, create one with "Other" option
    if response_schema is None:
        schema = {
            "type": "string",
            "enum": ["Other"],  # Single option that accepts free text
            "x-option-descriptions": {
                "Other": "Type your answer"
            }
        }
    else:
        schema = response_schema
    
    params = ElicitRequestFormParams(message=prompt, requestedSchema=schema)
    result = await ctx.handle_elicitation(params)
    # ... rest of the method ...
```

## Recommended Fix

**Option 1** is better because it:
1. Properly supports free-form text input at the provider level
2. Doesn't require hacky enum workarounds
3. Is more maintainable and clear about intent
4. Can be extended to support other input types in the future

## Testing

After implementing the fix, test with:

```python
async with ClaudeCodeAgent(...) as agent:
    async for event in agent.run_stream("Ask me a question using your question tool"):
        print(event)
```

The question should appear in the OpenCode UI and return the user's answer.
