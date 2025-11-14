# """Test ACP client-side skills discovery functionality."""

# from pathlib import Path
# import tempfile
# from textwrap import dedent
# from unittest.mock import AsyncMock

# import pytest

# from acp import ClientCapabilities, FileSystemCapability
# from llmling_agent import Agent
# from llmling_agent.delegation import AgentPool
# from llmling_agent.resource_providers.skills import SkillsResourceProvider
# from llmling_agent_server.acp_server.headless_client import HeadlessACPClient
# from llmling_agent_server.acp_server.session import ACPSession


# @pytest.fixture
# async def agent_pool():
#     """Create a real agent pool for testing."""

#     def simple_callback(message: str) -> str:
#         return f"Test response: {message}"

#     agent = Agent.from_callback(name="test_agent", callback=simple_callback)
#     pool = AgentPool()
#     pool.register("test_agent", agent)
#     async with pool:
#         yield pool


# async def test_client_skills_discovery_with_valid_skills(agent_pool):
#     """Test discovery of valid client-side skills."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Create .claude/skills directory structure
#         skills_dir = temp_path / ".claude" / "skills"
#         skills_dir.mkdir(parents=True)

#         # Create a test skill
#         test_skill_dir = skills_dir / "test_skill"
#         test_skill_dir.mkdir()

#         skill_content = dedent("""
#         ---
#         name: test_skill
#         description: A test skill for ACP integration
#         ---

#         # Test Skill Instructions

#         This is a test skill that demonstrates client-side skills discovery.

#         ## Usage

#         Use this skill when testing ACP skills functionality.
#         """).strip()

#         (test_skill_dir / "SKILL.md").write_text(skill_content)

#         # Create headless client
#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         # Set up session with file capabilities
#         fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
#         capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

#         session = ACPSession(
#             session_id="skills-test",
#             agent_pool=agent_pool,
#             current_agent_name="test_agent",
#             cwd=str(temp_path),
#             client=client,
#             acp_agent=AsyncMock(),
#             client_capabilities=capabilities,
#         )

#         # Get initial skills provider count
#         initial_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         # Run skills discovery
#         await session.init_client_skills()

#         # Check that client skills provider was added
#         final_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         # Should have one more provider (the client_skills provider)
#         assert len(final_providers) == len(initial_providers) + 1

#         # Find the client skills provider
#         client_skills_provider = next(
#             (p for p in final_providers if p.name == "client_skills"), None
#         )
#         assert client_skills_provider is not None

#         # Verify the skill was discovered
#         skills = await client_skills_provider.get_skills()
#         assert len(skills) == 1

#         skill = skills[0]
#         assert skill.name == "test_skill"
#         assert skill.description == "A test skill for ACP integration"
#         assert "Test Skill Instructions" in skill.instructions


# async def test_client_skills_no_skills_directory(agent_pool):
#     """Test behavior when .claude/skills directory doesn't exist."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Don't create .claude/skills directory

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
#         capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

#         session = ACPSession(
#             session_id="no-skills-test",
#             agent_pool=agent_pool,
#             current_agent_name="test_agent",
#             cwd=str(temp_path),
#             client=client,
#             acp_agent=AsyncMock(),
#             client_capabilities=capabilities,
#         )

#         # Get initial provider count
#         initial_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         # Run skills discovery (should not fail)
#         await session.init_client_skills()

#         # Should not add any new providers
#         final_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         assert len(final_providers) == len(initial_providers)

#         # Verify no client_skills provider was added
#         client_skills_provider = next(
#             (p for p in final_providers if p.name == "client_skills"), None
#         )
#         assert client_skills_provider is None


# async def test_client_skills_empty_directory(agent_pool):
#     """Test behavior when .claude/skills directory exists but is empty."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Create empty .claude/skills directory
#         skills_dir = temp_path / ".claude" / "skills"
#         skills_dir.mkdir(parents=True)

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
#         capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

#         session = ACPSession(
#             session_id="empty-skills-test",
#             agent_pool=agent_pool,
#             current_agent_name="test_agent",
#             cwd=str(temp_path),
#             client=client,
#             acp_agent=AsyncMock(),
#             client_capabilities=capabilities,
#         )

#         # Get initial provider count
#         initial_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         # Run skills discovery
#         await session.init_client_skills()

#         # Should not add any new providers since directory is empty
#         final_providers = [
#             p
#             for p in session.agent.tools.providers
#             if isinstance(p, SkillsResourceProvider)
#         ]

#         assert len(final_providers) == len(initial_providers)


# async def test_client_skills_multiple_skills(agent_pool):
#     """Test discovery of multiple client skills."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Create .claude/skills directory
#         skills_dir = temp_path / ".claude" / "skills"
#         skills_dir.mkdir(parents=True)

#         # Create first skill
#         skill1_dir = skills_dir / "skill_one"
#         skill1_dir.mkdir()
#         skill1_content = dedent("""
#         ---
#         name: skill_one
#         description: First test skill
#         ---

#         # First Skill
#         Instructions for the first skill.
#         """).strip()
#         (skill1_dir / "SKILL.md").write_text(skill1_content)

#         # Create second skill
#         skill2_dir = skills_dir / "skill_two"
#         skill2_dir.mkdir()
#         skill2_content = dedent("""
#         ---
#         name: skill_two
#         description: Second test skill
#         ---

#         # Second Skill
#         Instructions for the second skill.
#         """).strip()
#         (skill2_dir / "SKILL.md").write_text(skill2_content)

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
#         capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

#         session = ACPSession(
#             session_id="multi-skills-test",
#             agent_pool=agent_pool,
#             current_agent_name="test_agent",
#             cwd=str(temp_path),
#             client=client,
#             acp_agent=AsyncMock(),
#             client_capabilities=capabilities,
#         )

#         # Run skills discovery
#         await session.init_client_skills()

#         # Find the client skills provider
#         client_skills_provider = next(
#             (
#                 p
#                 for p in session.agent.tools.providers
#                 if isinstance(p, SkillsResourceProvider) and p.name == "client_skills"
#             ),
#             None,
#         )
#         assert client_skills_provider is not None

#         # Verify both skills were discovered
#         skills = await client_skills_provider.get_skills()
#         assert len(skills) == 2

#         skill_names = {skill.name for skill in skills}
#         assert skill_names == {"skill_one", "skill_two"}


# async def test_client_skills_invalid_skill_ignored(agent_pool):
#     """Test that invalid skills are ignored but valid ones are still processed."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Create .claude/skills directory
#         skills_dir = temp_path / ".claude" / "skills"
#         skills_dir.mkdir(parents=True)

#         # Create valid skill
#         valid_dir = skills_dir / "valid_skill"
#         valid_dir.mkdir()
#         valid_content = dedent("""
#         ---
#         name: valid_skill
#         description: A valid test skill
#         ---

#         # Valid Skill
#         This skill is properly formatted.
#         """).strip()
#         (valid_dir / "SKILL.md").write_text(valid_content)

#         # Create invalid skill (no frontmatter)
#         invalid_dir = skills_dir / "invalid_skill"
#         invalid_dir.mkdir()
#         invalid_content = "# Just a markdown file without frontmatter"
#         (invalid_dir / "SKILL.md").write_text(invalid_content)

#         # Create skill directory without SKILL.md file
#         empty_dir = skills_dir / "empty_skill"
#         empty_dir.mkdir()

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
#         capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

#         session = ACPSession(
#             session_id="mixed-skills-test",
#             agent_pool=agent_pool,
#             current_agent_name="test_agent",
#             cwd=str(temp_path),
#             client=client,
#             acp_agent=AsyncMock(),
#             client_capabilities=capabilities,
#         )

#         # Run skills discovery
#         await session.init_client_skills()

#         # Find the client skills provider
#         client_skills_provider = next(
#             (
#                 p
#                 for p in session.agent.tools.providers
#                 if isinstance(p, SkillsResourceProvider) and p.name == "client_skills"
#             ),
#             None,
#         )
#         assert client_skills_provider is not None

#         # Should only have the valid skill
#         skills = await client_skills_provider.get_skills()
#         assert len(skills) == 1
#         assert skills[0].name == "valid_skill"
