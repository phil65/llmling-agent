# AI-Human Interaction

This example demonstrates how AI agents can interact with humans:

- Using agent capabilities for human interaction
- Setting up a human agent in the pool
- Allowing AI to request human input when needed

## How It Works

1. We set up two agents:
   - An AI assistant with `can_ask_agents` capability
   - A human agent using the special "human" provider type

2. When the AI assistant encounters a question it can't answer:
   - It recognizes the need for human input
   - Uses its `can_ask_agents` capability to interact with the human agent
   - Incorporates the human's response into its answer

3. The conversation might look like this:
   ```
   Assistant: I need to check about Project DoomsDay's status. Let me ask the human.
   Human: Project DoomsDay is currently in Phase 2, with 60% completion.
   Assistant: Based on the human's input, Project DoomsDay is in Phase 2 and is 60% complete.
   ```

This demonstrates how to:

- Enable AI-human collaboration
- Control when AI can request human input
- Integrate human knowledge into AI responses
