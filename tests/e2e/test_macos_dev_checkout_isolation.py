"""The pytest suite must stay green on a normal developer macOS checkout.

20 tests fail on a developer macOS machine but pass in Linux CI. None are
product bugs — they are four test-isolation / platform-gating gaps. Each
test below drives one class **on Linux CI** by recreating the ambient
condition a developer machine has, so the suite guards the fix without
needing a Mac:

1. Linux-only bwrap/seccomp tests must not run on a macOS-like host —
   driven by simulating one (``sys.platform`` reads ``"darwin"`` and
   ``bwrap`` is absent) and asserting every affected test is skipped.
2. AF_UNIX socket paths built under ``tmp_path`` overflow the kernel's
   ``sun_path`` cap when the tmp dir is as long as macOS's ``$TMPDIR``
   (~48 bytes; macOS caps sockets at 104 bytes, Linux at 108) — driven
   with a macOS-length ``--basetemp``.
3. Ambient ``ANTHROPIC_DEFAULT_*_MODEL`` alias pins (a gateway-pinned
   developer shell) flip 4 tests — driven by exporting the pins.
4. Host state under ``$HOME`` that ``OMNIGENT_CONFIG_HOME`` isolation
   does not cover (``~/.codex/config.toml``, ``~/.databrickscfg``) flips
   spawn-env and runner tests — driven by pointing the child pytest's
   ``HOME`` at a polluted directory.

Not covered here: the macOS Keychain twist
(``tests/runtime/test_provider_spawn_env.py::test_detected_ambient_key_routes_with_no_config``)
— the ``claude auth status`` fallback is hard-gated to
``sys.platform == 'darwin'`` and reads a real Keychain, which no Linux
stand-in can reach.

Every test runs the *affected product tests* in a child pytest with the
polluting condition applied and asserts the correct outcome — red today,
green once the isolation/gating lands.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Class 1 — Linux-only tests that hard-fail off-Linux by design.
_BWRAP_RESOLVER_TESTS = (
    "tests/inner/test_bwrap_sandbox.py::test_resolve_default_keeps_cwd_read_only",
    "tests/inner/test_bwrap_sandbox.py::test_resolve_write_paths_dot_makes_cwd_writable",
    "tests/inner/test_bwrap_sandbox.py::test_resolve_default_cwd_allow_hidden_is_dot_venv",
    "tests/inner/test_bwrap_sandbox.py::test_resolve_explicit_cwd_allow_hidden_overrides_default",
)
_SECCOMP_CHILD_TESTS = (
    "tests/inner/test_seccomp.py::test_apply_baseline_denylist_blocks_ptrace_in_child",
    "tests/inner/test_seccomp.py::test_apply_baseline_denylist_does_not_break_subprocess_basics",
    "tests/inner/test_seccomp.py::test_arg_filter_blocks_socket_family_only",
    "tests/inner/test_seccomp.py::test_masked_eq_filter_blocks_clone_with_namespace_bit",
    "tests/inner/test_seccomp.py::test_unknown_syscall_silently_skipped",
)

# Class 2 — unix-socket paths built under pytest's tmp_path.
_AF_UNIX_TESTS = (
    "tests/inner/egress/test_proxy.py::test_proxy_start_unix",
    "tests/inner/test_terminal.py::test_server_survives_inner_process_exit_real_tmux",
)

# Class 3 — tests flipped by an ambient gateway alias pin.
_ALIAS_PIN_TESTS = (
    "tests/inner/test_claude_router_hook.py::test_rewrite_allows_with_routed_model",
    "tests/inner/test_claude_router_hook.py::test_legacy_task_tool_name_is_routed",
    "tests/inner/test_claude_router_hook.py::test_sdk_callback_maps_rewrite",
    "tests/runner/test_app_sessions_native_events_options.py::"
    "test_events_model_change_on_native_session_returns_503_when_bridge_not_ready",
)

# Class 4 — tests flipped by real provider state under $HOME.
_HOME_LEAK_TESTS = (
    "tests/runtime/test_openai_agents_sdk_spawn_env.py::test_no_model_produces_no_model_env_var",
    "tests/runtime/test_openai_agents_sdk_spawn_env.py::"
    "test_non_databricks_model_without_profile_omits_profile_env_var",
    "tests/runner/test_app_sessions_native_terminals_runtime.py::"
    "test_create_session_threads_workspace_to_pi_cwd",
)

# A codex config whose custom provider carries self-contained auth (the
# shape `isaac configure codex` writes): detection adopts it only when the
# base URL sits under a trusted Databricks parent domain and the table has
# an [auth] command — see codex_config_detection / is_databricks_ai_gateway_url.
_CODEX_CONFIG_TOML = """\
model_provider = "databricks"

[model_providers.databricks]
name = "Databricks"
base_url = "https://dogfood.cloud.databricks.com/ai-gateway/anthropic"

[model_providers.databricks.auth]
command = "printf FAKE-NOT-A-REAL-TOKEN"
"""

# Two profiles sharing one host — the databricks-sdk's host-based profile
# resolution then demands --profile, exactly the multi-profile state a
# Databricks developer machine has. Tokens are deliberately fake-shaped.
_DATABRICKSCFG = """\
[dogfood]
host = https://dogfood.cloud.databricks.com
token = FAKE-NOT-A-REAL-TOKEN-tests-only

[logfood-test-alias]
host = https://dogfood.cloud.databricks.com
token = FAKE-NOT-A-REAL-TOKEN-tests-only
"""

# macOS's per-user $TMPDIR (/var/folders/...) is ~48 bytes, so pytest's
# basetemp lands near 70 and tmp_path + socket filename overflows the
# 104-byte macOS sun_path cap. An 87-byte basetemp reproduces the same
# overflow against Linux's 108-byte cap.
_MACOS_LENGTH_BASETEMP_LEN = 87


def _run_pytest(
    node_ids: tuple[str, ...],
    *,
    extra_args: tuple[str, ...] = (),
    env_overrides: dict[str, str | None] | None = None,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    """Run the affected product tests in a child pytest, as a developer would.

    :param node_ids: Test node ids to run.
    :param extra_args: Extra pytest CLI args (e.g. ``--basetemp``).
    :param env_overrides: Env vars to set (or remove, with ``None``) on top
        of the current environment — the ambient developer-machine state.
    :param timeout: Child process timeout in seconds.
    :returns: The completed child process, output captured.
    """
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    for var, value in (env_overrides or {}).items():
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra_args, *node_ids],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    """Return the child's output tail for assertion messages."""
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "\n".join(out.strip().splitlines()[-30:])


def test_linux_only_suites_skip_on_macos_like_host() -> None:
    """Class 1: Linux-only bwrap/seccomp tests must skip on a macOS-like host.

    On macOS the 4 resolver tests die on ``resolve()``'s designed
    ``OSError`` ("linux_bwrap sandbox is only available on Linux") and the
    5 seccomp tests die in the forked child (no ``prctl`` symbol). Simulate
    the gating-relevant macOS conditions on Linux — ``sys.platform`` reads
    ``"darwin"`` and ``shutil.which("bwrap")`` finds nothing — and run the
    affected tests: each must be **skipped**. "passed" means no gate fired
    (the test ran; on a real Mac it would crash), "failed" is the crash a
    Mac developer sees. Gates of any legitimate shape satisfy this — a
    platform marker, a bwrap-availability marker, or a runtime skip.
    """
    node_ids = _BWRAP_RESOLVER_TESTS + _SECCOMP_CHILD_TESTS
    driver = textwrap.dedent(
        """
        import ctypes.util  # noqa: F401  (cache with real-platform behavior)
        import json
        import shutil
        import sys
        import urllib.request  # noqa: F401  (cache with real-platform behavior)

        import pytest

        _real_which = shutil.which

        def _which_without_bwrap(cmd, *args, **kwargs):
            if cmd == "bwrap":
                return None  # macOS has no bubblewrap
            return _real_which(cmd, *args, **kwargs)

        class DarwinSim:
            def pytest_sessionstart(self, session):
                # Patch after conftest imports (their import chains
                # platform-branch: psutil, urllib, ctypes.util) but before
                # test modules import during collection, so module-level
                # availability probes, skipif markers, and runtime platform
                # checks all see a macOS-like host.
                sys.platform = "darwin"
                shutil.which = _which_without_bwrap

        class Outcomes:
            def __init__(self):
                self.outcomes = {}

            def pytest_runtest_logreport(self, report):
                if report.skipped:
                    self.outcomes[report.nodeid] = "skipped"
                elif report.failed:
                    self.outcomes[report.nodeid] = "failed"
                elif report.when == "call" and report.passed:
                    self.outcomes[report.nodeid] = "passed"

        outcomes = Outcomes()
        node_ids = json.loads(sys.argv[1])
        pytest.main(
            ["-q", "-p", "no:cacheprovider", *node_ids],
            plugins=[DarwinSim(), outcomes],
        )
        print("OUTCOMES_JSON=" + json.dumps(outcomes.outcomes))
        sys.exit(0)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver, json.dumps(list(node_ids))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"macOS-simulation run did not complete:\n{_tail(proc)}"
    outcome_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("OUTCOMES_JSON=")
    )
    outcomes: dict[str, str] = json.loads(outcome_line[len("OUTCOMES_JSON=") :])

    not_skipped = {
        node_id: outcomes.get(node_id, "not run")
        for node_id in node_ids
        if outcomes.get(node_id) != "skipped"
    }
    assert not not_skipped, (
        "These Linux-only tests are not skipped on a macOS-like host "
        "(sys.platform == 'darwin', no bwrap), so a macOS checkout runs them "
        "and they hard-fail there ('passed' here means the gate never fired; "
        "'failed' is the crash a Mac developer sees):\n"
        + "\n".join(f"  {node_id}: {outcome}" for node_id, outcome in sorted(not_skipped.items()))
        + f"\n{_tail(proc)}"
    )


def test_af_unix_socket_tests_survive_macos_length_tmpdir() -> None:
    """Class 2: unix-socket tests must not overflow a macOS-length tmp dir.

    macOS caps ``sun_path`` at 104 bytes and its ``$TMPDIR`` is already
    ~48, so ``tmp_path`` (which embeds the test name) overflows before a
    socket filename is appended. Reproduced on Linux (108-byte cap) with
    an 87-byte ``--basetemp``. The repo's ``short_tmp_parent`` fixture
    is the existing cure; once these tests shorten their socket paths
    they pass at any basetemp length.
    """
    prefix = str(Path(tempfile.gettempdir()) / "omni-macos-tmpdir-sim-")
    padding = max(_MACOS_LENGTH_BASETEMP_LEN - len(prefix), 1)
    basetemp = prefix + "x" * padding

    node_ids = _AF_UNIX_TESTS
    if shutil.which("tmux") is None:
        node_ids = tuple(t for t in node_ids if "test_terminal" not in t)
        assert node_ids, "tmux missing and no tmux-free AF_UNIX test left to run"

    proc = _run_pytest(node_ids, extra_args=(f"--basetemp={basetemp}",))
    assert proc.returncode == 0, (
        "The unix-socket tests fail when pytest's tmp dir is as long as "
        "macOS's $TMPDIR (AF_UNIX sun_path overflow, class 2):\n"
        f"{_tail(proc)}"
    )


def test_alias_pin_tests_survive_ambient_gateway_pins() -> None:
    """Class 3: ambient ``ANTHROPIC_DEFAULT_*_MODEL`` pins must not flip tests.

    ``alias_pins()`` falls back to ``os.environ`` and
    ``claude_model_alias`` deliberately returns ``None`` on a mismatched
    pin, so any developer shell that pins these vars (anyone driving
    Claude through a gateway) turns 4 tests red. Per the report's bisect,
    any *single* pin alone is enough (a full coherent pin set can mask
    the bug), so pin exactly one. The affected tests must isolate
    themselves from the ambient pins.
    """
    proc = _run_pytest(
        _ALIAS_PIN_TESTS,
        env_overrides={
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "databricks-claude-opus-5",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": None,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": None,
        },
    )
    assert proc.returncode == 0, (
        "Ambient ANTHROPIC_DEFAULT_*_MODEL pins (a gateway-pinned developer "
        "shell) flip these tests (class 3):\n"
        f"{_tail(proc)}"
    )


def test_home_state_tests_survive_polluted_home(tmp_path: Path) -> None:
    """Class 4: provider state under ``$HOME`` must not flip isolated tests.

    Several suites isolate ``OMNIGENT_CONFIG_HOME`` but ambient provider
    detection also reads ``~/.codex/config.toml`` and ``~/.databrickscfg``
    — which live under ``HOME``. A normal Databricks developer machine has
    both, so these tests go red there. The affected tests must isolate
    ``HOME`` (or stub detection at the right level).
    """
    polluted_home = tmp_path / "developer-home"
    (polluted_home / ".codex").mkdir(parents=True)
    (polluted_home / ".codex" / "config.toml").write_text(_CODEX_CONFIG_TOML)
    (polluted_home / ".databrickscfg").write_text(_DATABRICKSCFG)

    proc = _run_pytest(
        _HOME_LEAK_TESTS,
        env_overrides={
            "HOME": str(polluted_home),
            # A developer machine may or may not pin these; keep the child
            # deterministic so only the $HOME files drive the outcome.
            "DATABRICKS_CONFIG_PROFILE": None,
        },
    )
    assert proc.returncode == 0, (
        "Real ~/.codex/config.toml + ~/.databrickscfg under $HOME leak into "
        "OMNIGENT_CONFIG_HOME-isolated tests (class 4):\n"
        f"{_tail(proc)}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
