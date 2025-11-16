"""CLI documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("CLI")

CLI_PATH = "llmling_agent.__main__:cli"


@nav.route.page(is_index=True, hide="toc")
def _(page: mk.MkPage) -> None:
    # page += mk.MkBinaryImage.for_file("docs/assets/cli.gif")
    # page += mk.MkTemplate("cli_index.jinja")
    page += mk.MkTemplate("docs/cli.md")


@nav.route.page("add", icon="mdi:plus-circle")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="add")


@nav.route.page("run", icon="mdi:play")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="run")


@nav.route.page("list", icon="mdi:format-list-bulleted")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="list")


@nav.route.page("set", icon="mdi:cog")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="set")


@nav.route.page("chat", icon="mdi:chat")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="chat")


@nav.route.page("quickstart", icon="mdi:rocket")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="quickstart")


@nav.route.page("task", icon="mdi:clipboard-check")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="task")


@nav.route.page("watch", icon="mdi:eye")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="watch")


@nav.route.page("history", icon="mdi:history")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="history")


@nav.route.page("launch", icon="mdi:launch")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="launch")


@nav.route.page("web", icon="mdi:web")
def _(page: mk.MkPage) -> None:
    page += mk.MkCliDoc(CLI_PATH, prog_name="web")
