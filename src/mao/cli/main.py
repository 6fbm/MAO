"""Command line entry point (``mao`` / ``python -m mao``)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mao import __version__

if TYPE_CHECKING:
    from mao.app import AppContext
    from mao.cli.console import ConsoleUI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mao",
        description="Multi AI Orchestrator - several AI models work as a team of agents in the terminal.",
    )
    parser.add_argument("--home", help="Program folder holding config/ and logs/ (default: MAO_HOME or the installation folder)")
    parser.add_argument("--workspace", "-w", help="Set the workspace for this run only")
    parser.add_argument("--demo", action="store_true", help="Offline demo: simulated model replies, real tools, files and tests")
    parser.add_argument("--plain", action="store_true", help="No live interface, line-by-line output")
    parser.add_argument("--debug", action="store_true", help="Show detailed events")
    parser.add_argument("--yes", "-y", action="store_true", help="Confirm and run plans automatically")
    parser.add_argument("--no-discover", action="store_true", help="Do not detect local models automatically at start")
    parser.add_argument("-c", "--command", action="append", default=[], metavar="COMMAND", help="Run a command (can be repeated), then exit")
    parser.add_argument("--version", action="version", version=f"mao {__version__}")
    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")
    init = sub.add_parser("init", help="Create the configuration files")
    init.add_argument("--force", action="store_true", help="overwrite existing files with the templates")
    sub.add_parser("doctor", help="Check the environment, the configuration and the providers")
    sub.add_parser("sessions", help="List the stored sessions")
    plan = sub.add_parser("plan", help="Plan a task and run it after confirmation")
    plan.add_argument("task", nargs="+", help="Description of the task")
    run = sub.add_parser("run", help="Run/resume the plan of a stored session (default: the last one)")
    run.add_argument("session", nargs="?", help="Session-ID")
    demo = sub.add_parser("demo-workspace", help="Create an example project with a deliberate bug for the demo")
    demo.add_argument("path", help="Target directory")
    return parser


def _setup_console() -> tuple[int, int] | None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if sys.platform != "win32" or not sys.stdout.isatty():
        return None
    import ctypes

    kernel32 = ctypes.windll.kernel32
    previous = (int(kernel32.GetConsoleOutputCP()), int(kernel32.GetConsoleCP()))
    if previous[0] == 0:
        return None
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)
    return previous


def _restore_console(previous: tuple[int, int] | None) -> None:
    if previous is None:
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(previous[0])
    kernel32.SetConsoleCP(previous[1])


def _install_interrupt_handler(ui: ConsoleUI) -> None:
    def handler(signum: int, frame: object) -> None:
        # first Ctrl+C pauses the running task; a quick second one aborts the program
        if ui.interrupts.trigger():
            signal.default_int_handler(signum, frame)

    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGINT, handler)


def _exit_code(app: AppContext) -> int:
    result = app.orchestrator.last_result
    if result is None:
        return 0
    return 0 if result.status == "completed" else 1


def _init_config(args: argparse.Namespace) -> int:
    from mao.config.loader import ConfigManager
    from mao.models import local as local_models
    from mao.paths import AppPaths

    paths = AppPaths.discover(args.home)
    written = ConfigManager(paths.config_dir).ensure_templates(overwrite=args.force)
    print(f"Configuration folder: {paths.config_dir}")
    for path in written:
        print(f"  created: {path.name}")
    if not written:
        print("  All files are already present (use --force to replace them with the templates).")
    print(f"Model folder:         {local_models.ensure_dir(paths.models_dir)}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.text import Text

    from mao.app import AppContext
    from mao.cli.console import ConsoleUI
    from mao.cli.repl import Repl
    from mao.paths import AppPaths

    paths = AppPaths.discover(args.home)
    interactive = bool(sys.stdin.isatty() and sys.stdout.isatty() and not args.plain)
    console = Console(highlight=False, markup=False, emoji=False)
    with console.status("Starting Multi AI Orchestrator …"):
        app = await AppContext.create(paths, demo=args.demo, discover=not args.no_discover)
    try:
        if args.workspace:
            workspace = Path(args.workspace).expanduser()
            if not workspace.is_dir():
                console.print(Text(f"Workspace does not exist: {workspace}", style="bold red"))
                return 2
            app.config.settings.workspace = str(workspace.resolve())
        ui = ConsoleUI(app, interactive=interactive, debug=args.debug)
        app.set_handlers(ui.approval_handler, ui.limit_handler)
        _install_interrupt_handler(ui)
        repl = Repl(app, ui, auto_approve_plan=args.yes)

        if args.subcommand == "doctor":
            await repl.dispatch("/doctor")
            return 0
        if args.subcommand == "sessions":
            await repl.dispatch("/sessions")
            return 0
        if args.subcommand == "plan":
            await repl.dispatch("/plan " + " ".join(args.task))
        elif args.subcommand == "run":
            session_id = args.session
            if session_id is None:
                latest = app.store.latest()
                if latest is None:
                    ui.error("No sessions available.")
                    return 1
                session_id = latest.id
            await repl.dispatch(f"/run {session_id}")
        elif args.command:
            for line in args.command:
                await repl.dispatch(line)
                if repl.should_exit:
                    break
        else:
            await repl.run()
        if repl.background is not None and not repl.background.done():
            ui.warn("A paused task is stopped in a controlled way on exit (resumable with /run <id>).")
            with contextlib.suppress(Exception):
                app.orchestrator.stop("Program exited")
                await ui.attach(repl.background)
        return _exit_code(app)
    finally:
        await app.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "init":
        return _init_config(args)
    if args.subcommand == "demo-workspace":
        from mao.demo import create_demo_workspace

        target = create_demo_workspace(Path(args.path).expanduser().resolve())
        print(f"Demo workspace created: {target}")
        print(f'Start it with:  mao --demo --workspace "{target}"')
        return 0

    from mao.core.errors import MaoError

    previous = _setup_console()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except MaoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        _restore_console(previous)


if __name__ == "__main__":
    raise SystemExit(main())
