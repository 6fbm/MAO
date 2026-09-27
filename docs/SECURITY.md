# Security model

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use the **Security** tab of this
repository and the **Report a vulnerability** button (GitHub Security Advisories) instead. You can
normally expect a first reply within 7 days.

This is a hobby project maintained by one person, with no support commitment: there is no bug bounty
and no guaranteed response time. What is described here is the model that is actually implemented — it
is not a promise of complete isolation (see [LIMITATIONS.md](LIMITATIONS.md)).

Agents are language models: they can be wrong, and they can be manipulated by the contents of files and
web pages (prompt injection). The system therefore treats every agent action as untrusted and checks it
technically, regardless of what the prompt says.

## Layers

1. **Tool selection**: an agent only sees tools from its tool groups that it has permissions for. In
   PLAN mode, writing tools are not offered at all.
2. **Permission check in the executor**: even if a model calls a tool that was not offered, the call is
   rejected (permissions, mode, argument schema).
3. **Workspace sandbox**: paths are resolved with `realpath` (symlinks, junctions) and must lie inside
   the workspace. Rejected are UNC and device paths, `~`, alternate data streams (`file:stream`) and
   reserved Windows names (`CON`, `NUL`, …). `.git` is not writable. The root directory itself cannot be
   deleted.
4. **Risk assessment of commands**: a denylist (always blocked) plus rules for critical/high/medium risk
   (formatting, registry, deletion, destructive git commands, downloads piped into execution, encoded
   PowerShell, permission changes, package installs, redirections, paths outside the workspace).
   Programs are also checked by their base name so that absolute paths cannot bypass a rule. Chained
   commands count as medium risk at minimum and are never allowlisted.
5. **Approvals**: rules per kind of action; one prompt at a time; "always" applies to the session only
   and never to critical actions; decisions are recorded in `tools.log`.
6. **Process isolation (lightweight)**: stdin closed, timeouts that kill the process tree, output capped,
   environment variables holding keys/tokens/passwords removed.
7. **Network**: `fetch_url` allows http/https only, resolves the host and blocks private and local
   addresses — after every redirect as well.
8. **Secrets**: DPAPI-encrypted (bound to the Windows user) or environment variables; displayed only as a
   fingerprint; a global redactor covers logs, events, messages, the blackboard and tool output (known
   keys exactly, plus patterns of common key formats). The HTTP libraries do not log requests.
9. **Traceability and a way back**: backups of every changed or deleted file per session, `/diff`,
   `/rollback`, git checkpoints (`refs/mao/checkpoints/…`), complete event and tool logs.

## Recommendations

- For unknown or sensitive projects, set `permissions.ceiling` restrictively (e.g. `terminal: false`).
- Only use `auto_approve` in throwaway environments.
- For real isolation, run tasks in a VM, a container or the Windows Sandbox: commands run with the
  permissions of the logged-in user.
- Set `privacy.local_only: true` when code must not leave the machine.
- Sensitive files (`.env`, keys) are only read after approval — whatever an agent reads goes to the
  model behind that agent.
