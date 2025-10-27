from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, ImportString, SecretStr
from schemez import Schema


if TYPE_CHECKING:
    from llmling_models import CostOptimizedMultiModel, DelegationMultiModel, InputModel
    from llmling_models.augmented import AugmentedModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.function import FunctionModel


class BaseModelConfig(Schema):
    """Base for model configurations."""

    type: str = Field(init=False)
    """Type discriminator for model configs."""

    def get_model(self) -> Any:
        """Create and return actual model instance."""
        msg = f"Model creation not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)


class PrePostPromptConfig(Schema):
    """Configuration for pre/post prompts."""

    text: str
    """The prompt text to be applied."""

    model: str
    """The model to use for processing the prompt."""


class AugmentedModelConfig(BaseModelConfig):
    """Configuration for model with pre/post prompt processing."""

    type: Literal["augmented"] = Field(default="augmented", init=False)
    """Type identifier for augmented model."""

    main_model: str
    """The primary model identifier."""

    pre_prompt: PrePostPromptConfig | None = None
    """Optional configuration for prompt preprocessing."""

    post_prompt: PrePostPromptConfig | None = None
    """Optional configuration for prompt postprocessing."""

    def get_model(self) -> AugmentedModel:
        from llmling_models.augmented import AugmentedModel

        return AugmentedModel(
            main_model=self.main_model,  # type: ignore
            pre_prompt=self.pre_prompt,  # type: ignore
            post_prompt=self.post_prompt,  # type: ignore
        )


class CostOptimizedModelConfig(BaseModelConfig):
    """Configuration for cost-optimized model selection."""

    type: Literal["cost-optimized"] = Field(default="cost-optimized", init=False)
    """Type identifier for cost-optimized model."""

    models: list[str | BaseModelConfig] = Field(min_length=1)
    """List of available models to choose from."""

    max_input_cost: float = Field(gt=0)
    """Maximum cost threshold for input processing."""

    strategy: Literal["cheapest_possible", "best_within_budget"] = "best_within_budget"
    """Strategy for model selection based on cost."""

    def get_model(self) -> CostOptimizedMultiModel:
        from llmling_models.multimodels import CostOptimizedMultiModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return CostOptimizedMultiModel(
            models=converted_models,
            max_input_cost=self.max_input_cost,
            strategy=self.strategy,
        )


class DelegationModelConfig(BaseModelConfig):
    """Configuration for delegation-based model selection."""

    type: Literal["delegation"] = Field(default="delegation", init=False)
    """Type identifier for delegation model."""

    selector_model: str | BaseModelConfig
    """Model responsible for selecting which model to use."""

    models: list[str | BaseModelConfig] = Field(min_length=1)
    """List of available models to choose from."""

    selection_prompt: str
    """Prompt used to guide the selector model's decision."""

    model_descriptions: dict[str, str] | None = None
    """Optional descriptions of each model for selection purposes."""

    def get_model(self) -> DelegationMultiModel:
        from llmling_models.multimodels import DelegationMultiModel

        # Convert selector if it's a config
        selector = (
            self.selector_model.get_model()
            if isinstance(self.selector_model, BaseModelConfig)
            else self.selector_model
        )

        # Convert model list
        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]

        return DelegationMultiModel(
            selector_model=selector,
            models=converted_models,
            selection_prompt=self.selection_prompt,
        )


class FallbackModelConfig(BaseModelConfig):
    """Configuration for fallback strategy."""

    type: Literal["fallback"] = Field(default="fallback", init=False)
    """Type identifier for fallback model."""

    models: list[str | BaseModelConfig] = Field(min_length=1)
    """Ordered list of models to try in sequence."""

    def get_model(self) -> FallbackModel:
        from pydantic_ai.models.fallback import FallbackModel

        # Convert nested configs to models
        converted_models = [
            model.get_model() if isinstance(model, BaseModelConfig) else model
            for model in self.models
        ]
        return FallbackModel(*converted_models)  # type: ignore


class ImportModelConfig(BaseModelConfig):
    """Configuration for importing external models."""

    type: Literal["import"] = Field(default="import", init=False)
    """Type identifier for import model."""

    model: ImportString
    """Import path to the model class or function."""

    kw_args: dict[str, str] = Field(default_factory=dict)
    """Keyword arguments to pass to the imported model."""

    def get_model(self) -> Any:
        return self.model(**self.kw_args) if isinstance(self.model, type) else self.model


class InputModelConfig(BaseModelConfig):
    """Configuration for human input model."""

    type: Literal["input"] = Field(default="input", init=False)
    """Type identifier for input model."""

    prompt_template: str = Field(default="👤 Please respond to: {prompt}")
    """Template for displaying the prompt to the user."""

    show_system: bool = Field(default=True)
    """Whether to show system messages."""

    input_prompt: str = Field(default="Your response: ")
    """Text displayed when requesting input."""

    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
    )
    """Handler for processing user input."""

    def get_model(self) -> InputModel:
        from llmling_models.inputmodel import InputModel

        return InputModel(
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class RemoteInputConfig(BaseModelConfig):
    """Configuration for remote human input."""

    type: Literal["remote-input"] = Field(default="remote-input", init=False)
    """Type identifier for remote input model."""

    url: str = "ws://localhost:8000/v1/chat/stream"
    """WebSocket URL for connecting to the remote input service."""

    api_key: SecretStr | None = None
    """Optional API key for authentication."""

    def get_model(self) -> Any:
        from llmling_models.remote_input.client import RemoteInputModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteInputModel(url=self.url, api_key=key)


class RemoteProxyConfig(BaseModelConfig):
    """Configuration for remote model proxy."""

    type: Literal["remote-proxy"] = Field(default="remote-proxy", init=False)
    """Type identifier for remote proxy model."""

    url: str = "ws://localhost:8000/v1/completion/stream"
    """WebSocket URL for connecting to the remote model service."""

    api_key: SecretStr | None = None
    """Optional API key for authentication."""

    def get_model(self) -> Any:
        from llmling_models.remote_model.client import RemoteProxyModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteProxyModel(url=self.url, api_key=key)


class TokenOptimizedModelConfig(BaseModelConfig):
    """Configuration for token-optimized model selection."""

    type: Literal["token-optimized"] = Field(default="token-optimized", init=False)
    """Type identifier for token-optimized model."""

    models: list[str | BaseModelConfig] = Field(min_length=1)
    """List of available models to choose from based on token optimization."""

    strategy: Literal["efficient", "maximum_context"] = Field(default="efficient")
    """Strategy for selecting models based on token usage."""

    def get_model(self) -> Any:
        from llmling_models.multimodels import TokenOptimizedMultiModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return TokenOptimizedMultiModel(
            models=converted_models,
            strategy=self.strategy,
        )


class UserSelectModelConfig(BaseModelConfig):
    """Configuration for interactive model selection."""

    type: Literal["user-select"] = Field(default="user-select", init=False)
    """Type identifier for user-select model."""

    models: list[str | BaseModelConfig] = Field(min_length=1)
    """List of models the user can choose from."""

    prompt_template: str = Field(default="🤖 Choose a model for: {prompt}")
    """Template for displaying the choice prompt to the user."""

    show_system: bool = Field(default=True)
    """Whether to show system messages during selection."""

    input_prompt: str = Field(default="Enter model number (0-{max}): ")
    """Text displayed when requesting model selection."""

    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
    )
    """Handler for processing user selection input."""

    def get_model(self) -> Any:
        from llmling_models.multimodels import UserSelectModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return UserSelectModel(
            models=converted_models,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class StringModelConfig(BaseModelConfig):
    """Configuration for string-based model references."""

    type: Literal["string"] = Field(default="string", init=False)
    """Type identifier for string model."""

    identifier: str
    """String identifier for the model."""

    def get_model(self) -> Any:
        from llmling_models.model_types import StringModel

        return StringModel(identifier=self.identifier)


class FunctionModelConfig(BaseModelConfig):
    """Configuration for function-based model references."""

    type: Literal["function"] = Field(default="function", init=False)
    """Type identifier for function model."""

    function: ImportString
    """Function identifier for the model."""

    def get_model(self) -> FunctionModel:
        from llmling_models import function_to_model

        return function_to_model(self.function)


class TestModelConfig(BaseModelConfig):
    """Configuration for test models."""

    type: Literal["test"] = Field(default="test", init=False)
    """Type identifier for test model."""

    custom_output_text: str | None = None
    """Optional custom text to return from the test model."""

    call_tools: list[str] | Literal["all"] = "all"
    """Tools that can be called by the test model."""

    def get_model(self) -> Any:
        from pydantic_ai.models.test import TestModel

        return TestModel(
            custom_output_text=self.custom_output_text,
            call_tools=self.call_tools,
        )


AnyModelConfig = Annotated[
    AugmentedModelConfig
    | CostOptimizedModelConfig
    | DelegationModelConfig
    | FallbackModelConfig
    | FunctionModelConfig
    | ImportModelConfig
    | InputModelConfig
    | RemoteInputConfig
    | RemoteProxyConfig
    | TokenOptimizedModelConfig
    | StringModelConfig
    | TestModelConfig
    | UserSelectModelConfig,
    Field(discriminator="type"),
]
