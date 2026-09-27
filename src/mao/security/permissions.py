"""Capabilities required by tools and resolution of agent permissions."""

from __future__ import annotations

from enum import Enum

from mao.config.schema import PermissionOverrides, PermissionSet


class Capability(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    INTERNET = "internet"
    GIT_READ = "git_read"
    GIT_WRITE = "git_write"
    CONSULT = "consult"


CAPABILITY_LABELS: dict[Capability, str] = {
    Capability.READ: "Read files",
    Capability.WRITE: "Write files",
    Capability.DELETE: "Delete files",
    Capability.EXECUTE: "Run commands",
    Capability.INTERNET: "Internet access",
    Capability.GIT_READ: "Read git",
    Capability.GIT_WRITE: "Write git",
    Capability.CONSULT: "Consult other agents",
}


def granted_capabilities(perms: PermissionSet) -> set[Capability]:
    caps: set[Capability] = set()
    if perms.read:
        caps.add(Capability.READ)
    if perms.write:
        caps.add(Capability.WRITE)
    if perms.delete:
        caps.add(Capability.DELETE)
    if perms.execute:
        caps.add(Capability.EXECUTE)
    if perms.internet:
        caps.add(Capability.INTERNET)
    if perms.git in ("read", "write"):
        caps.add(Capability.GIT_READ)
    if perms.git == "write":
        caps.add(Capability.GIT_WRITE)
    if perms.consult:
        caps.add(Capability.CONSULT)
    return caps


def missing_capabilities(perms: PermissionSet, required: frozenset[Capability] | set[Capability]) -> set[Capability]:
    return set(required) - granted_capabilities(perms)


def resolve_permissions(
    defaults: PermissionOverrides | None,
    role: PermissionOverrides | None,
    agent: PermissionOverrides | None,
    ceiling: PermissionOverrides | None = None,
) -> PermissionSet:
    """defaults → role → agent (later wins), then the global ceiling may only reduce."""
    perms = PermissionSet()
    for layer in (defaults, role, agent):
        if layer is not None:
            perms = layer.apply(perms)
    if ceiling is not None:
        perms = ceiling.restrict(perms)
    return perms
