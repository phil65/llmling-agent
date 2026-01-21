"""Mode categories for ACPAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from agentpool.agents.modes import ConfigOptionChanged, ModeCategoryProtocol, ModeInfo


if TYPE_CHECKING:
    from acp.schema import SessionConfigOption, SessionModelState, SessionModeState
    from agentpool.agents.acp_agent.acp_agent import ACPAgent


# =============================================================================
# Mode category implementations
# =============================================================================


@dataclass
class ACPModeCategory(ModeCategoryProtocol["ACPAgent"]):
    """Mode category for ACP - handles permissions/approval policy.

    Automatically uses config_options API if available, falls back to
    set_session_mode for older servers.
    """

    available_modes: list[ModeInfo]

    id: ClassVar[str] = "mode"
    name: ClassVar[str] = "Mode"
    category: ClassVar[str] = "mode"

    def get_current(self, agent: ACPAgent) -> str:
        """Get current mode from ACP state."""
        # Try config_options first
        if agent._state and agent._state.config_options:
            for opt in agent._state.config_options:
                if opt.id == "mode":
                    return opt.current_value
        # Fall back to legacy modes state
        if agent._state and agent._state.modes:
            return agent._state.modes.current_mode_id
        return ""

    async def apply(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply mode change - uses config_options if available, else legacy API."""
        valid_ids = {m.id for m in self.available_modes}
        if mode_id not in valid_ids:
            msg = f"Invalid mode '{mode_id}' for category '{self.id}'. Valid: {valid_ids}"
            raise ValueError(msg)

        if not agent._connection or not agent._sdk_session_id or not agent._state:
            raise RuntimeError("Not connected to ACP server")

        # Try config_options API first
        if agent._state.config_options:
            await self._apply_via_config_options(agent, mode_id)
        else:
            await self._apply_via_legacy(agent, mode_id)

        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)

    async def _apply_via_config_options(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply using new config_options API."""
        from acp.schema import SetSessionConfigOptionRequest

        assert agent._connection is not None
        assert agent._state is not None

        config_request = SetSessionConfigOptionRequest(
            session_id=agent._sdk_session_id,
            config_id="mode",
            value=mode_id,
        )
        response = await agent._connection.set_session_config_option(config_request)

        if response.config_options:
            agent._state.config_options = list(response.config_options)

        agent.log.info("Mode changed via config_options", mode_id=mode_id)

    async def _apply_via_legacy(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply using legacy set_session_mode API."""
        from acp.schema import SetSessionModeRequest

        assert agent._connection is not None
        assert agent._state is not None

        mode_request = SetSessionModeRequest(session_id=agent._sdk_session_id, mode_id=mode_id)
        await agent._connection.set_session_mode(mode_request)

        if agent._state.modes:
            agent._state.modes.current_mode_id = mode_id

        agent.log.info("Mode changed via legacy API", mode_id=mode_id)

    @classmethod
    def from_state(
        cls,
        config_options: list[SessionConfigOption] | None,
        modes_state: SessionModeState | None,
    ) -> ACPModeCategory | None:
        """Create from ACP state - prefers config_options, falls back to modes_state.

        Args:
            config_options: SessionConfigOption list (new API)
            modes_state: SessionModeState (legacy API)

        Returns:
            ACPModeCategory instance, or None if no mode info available
        """
        from acp.schema import SessionConfigSelectGroup

        # Try config_options first
        if config_options:
            for config_opt in config_options:
                if config_opt.id != "mode":
                    continue
                mode_infos: list[ModeInfo] = []
                if isinstance(config_opt.options, list):
                    for opt_item in config_opt.options:
                        if isinstance(opt_item, SessionConfigSelectGroup):
                            mode_infos.extend(
                                ModeInfo(
                                    id=sub_opt.value,
                                    name=sub_opt.name,
                                    description=sub_opt.description or "",
                                    category_id="mode",
                                )
                                for sub_opt in opt_item.options
                            )
                        else:
                            mode_infos.append(
                                ModeInfo(
                                    id=opt_item.value,
                                    name=opt_item.name,
                                    description=opt_item.description or "",
                                    category_id="mode",
                                )
                            )
                return cls(available_modes=mode_infos)

        # Fall back to legacy modes state
        if modes_state:
            modes = [
                ModeInfo(
                    id=m.id,
                    name=m.name,
                    description=m.description or "",
                    category_id="mode",
                )
                for m in modes_state.available_modes
            ]
            return cls(available_modes=modes)

        return None


@dataclass
class ACPModelCategory(ModeCategoryProtocol["ACPAgent"]):
    """Model category for ACP - handles model selection.

    Automatically uses config_options API if available, falls back to
    set_session_model for older servers.
    """

    available_modes: list[ModeInfo]

    id: ClassVar[str] = "model"
    name: ClassVar[str] = "Model"
    category: ClassVar[str] = "model"

    def get_current(self, agent: ACPAgent) -> str:
        """Get current model from ACP state."""
        # Try config_options first
        if agent._state and agent._state.config_options:
            for opt in agent._state.config_options:
                if opt.id == "model":
                    return opt.current_value
        # Fall back to legacy models state
        if agent._state and agent._state.models:
            return agent._state.models.current_model_id
        return ""

    async def apply(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply model change - uses config_options if available, else legacy API."""
        valid_ids = {m.id for m in self.available_modes}
        if mode_id not in valid_ids:
            msg = f"Invalid model '{mode_id}' for category '{self.id}'. Valid: {valid_ids}"
            raise ValueError(msg)

        if not agent._connection or not agent._sdk_session_id or not agent._state:
            raise RuntimeError("Not connected to ACP server")

        # Try config_options API first
        if agent._state.config_options:
            await self._apply_via_config_options(agent, mode_id)
        else:
            await self._apply_via_legacy(agent, mode_id)

        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)

    async def _apply_via_config_options(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply using new config_options API."""
        from acp.schema import SetSessionConfigOptionRequest

        assert agent._connection is not None
        assert agent._state is not None

        config_request = SetSessionConfigOptionRequest(
            session_id=agent._sdk_session_id,
            config_id="model",
            value=mode_id,
        )
        response = await agent._connection.set_session_config_option(config_request)

        if response.config_options:
            agent._state.config_options = list(response.config_options)

        agent.log.info("Model changed via config_options", model_id=mode_id)

    async def _apply_via_legacy(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply using legacy set_session_model API."""
        from acp.schema import SetSessionModelRequest

        assert agent._connection is not None
        assert agent._state is not None

        request = SetSessionModelRequest(session_id=agent._sdk_session_id, model_id=mode_id)
        if await agent._connection.set_session_model(request):
            agent._state.current_model_id = mode_id
            agent.log.info("Model changed via legacy API", model_id=mode_id)
        else:
            msg = (
                "Remote ACP agent does not support model changes. "
                "set_session_model returned no response."
            )
            raise RuntimeError(msg)

    @classmethod
    def from_state(
        cls,
        config_options: list[SessionConfigOption] | None,
        models_state: SessionModelState | None,
    ) -> ACPModelCategory | None:
        """Create from ACP state - prefers config_options, falls back to models_state.

        Args:
            config_options: SessionConfigOption list (new API)
            models_state: SessionModelState (legacy API)

        Returns:
            ACPModelCategory instance, or None if no model info available
        """
        from acp.schema import SessionConfigSelectGroup

        # Try config_options first
        if config_options:
            for config_opt in config_options:
                if config_opt.id != "model":
                    continue
                mode_infos: list[ModeInfo] = []
                if isinstance(config_opt.options, list):
                    for opt_item in config_opt.options:
                        if isinstance(opt_item, SessionConfigSelectGroup):
                            mode_infos.extend(
                                ModeInfo(
                                    id=sub_opt.value,
                                    name=sub_opt.name,
                                    description=sub_opt.description or "",
                                    category_id="model",
                                )
                                for sub_opt in opt_item.options
                            )
                        else:
                            mode_infos.append(
                                ModeInfo(
                                    id=opt_item.value,
                                    name=opt_item.name,
                                    description=opt_item.description or "",
                                    category_id="model",
                                )
                            )
                return cls(available_modes=mode_infos)

        # Fall back to legacy models state
        if models_state:
            models = [
                ModeInfo(
                    id=m.model_id,
                    name=m.name,
                    description=m.description or "",
                    category_id="model",
                )
                for m in models_state.available_models
            ]
            return cls(available_modes=models)

        return None


@dataclass
class ACPGenericCategory(ModeCategoryProtocol["ACPAgent"]):
    """Generic category for ACP - handles any config_option that's not mode/model.

    For things like thought_level or custom categories that only exist in
    the config_options API.
    """

    id: str
    name: str
    available_modes: list[ModeInfo]
    category: str | None = None

    def get_current(self, agent: ACPAgent) -> str:
        """Get current value from ACP state."""
        if agent._state and agent._state.config_options:
            for opt in agent._state.config_options:
                if opt.id == self.id:
                    return opt.current_value
        return ""

    async def apply(self, agent: ACPAgent, mode_id: str) -> None:
        """Apply config option change via config_options API."""
        from acp.schema import SetSessionConfigOptionRequest

        valid_ids = {m.id for m in self.available_modes}
        if mode_id not in valid_ids:
            msg = f"Invalid value '{mode_id}' for category '{self.id}'. Valid: {valid_ids}"
            raise ValueError(msg)

        if not agent._connection or not agent._sdk_session_id or not agent._state:
            raise RuntimeError("Not connected to ACP server")

        if not agent._state.config_options:
            raise RuntimeError(f"Server does not support config_options, cannot set {self.id!r}")

        config_request = SetSessionConfigOptionRequest(
            session_id=agent._sdk_session_id,
            config_id=self.id,
            value=mode_id,
        )
        response = await agent._connection.set_session_config_option(config_request)

        if response.config_options:
            agent._state.config_options = list(response.config_options)

        agent.log.info("Config option changed", config_id=self.id, value=mode_id)
        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)

    @classmethod
    def from_config_options(
        cls,
        config_options: list[SessionConfigOption],
    ) -> list[ACPGenericCategory]:
        """Create categories for non-mode/model config options.

        Args:
            config_options: SessionConfigOption list

        Returns:
            List of ACPGenericCategory for options that aren't mode/model
        """
        from acp.schema import SessionConfigSelectGroup

        categories: list[ACPGenericCategory] = []

        for config_opt in config_options:
            # Skip mode and model - they have dedicated categories
            if config_opt.id in ("mode", "model"):
                continue

            mode_infos: list[ModeInfo] = []
            if isinstance(config_opt.options, list):
                for opt_item in config_opt.options:
                    if isinstance(opt_item, SessionConfigSelectGroup):
                        mode_infos.extend(
                            ModeInfo(
                                id=sub_opt.value,
                                name=sub_opt.name,
                                description=sub_opt.description or "",
                                category_id=config_opt.id,
                            )
                            for sub_opt in opt_item.options
                        )
                    else:
                        mode_infos.append(
                            ModeInfo(
                                id=opt_item.value,
                                name=opt_item.name,
                                description=opt_item.description or "",
                                category_id=config_opt.id,
                            )
                        )

            categories.append(
                cls(
                    id=config_opt.id,
                    name=config_opt.name,
                    available_modes=mode_infos,
                    category=config_opt.category or "other",
                )
            )

        return categories
