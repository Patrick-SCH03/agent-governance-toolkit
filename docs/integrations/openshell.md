---
title: OpenShell integration
last_reviewed: 2026-09-23
owner: docs-team
---

# OpenShell integration

> **Status:** The OpenShell-specific `GovernanceSkill` library was retired in the
> v5 ACS migration. This page preserves the migration path for existing
> integrations; it does not describe a supported OpenShell-specific adapter.

## Govern an OpenShell-hosted agent

Use the Agent Control Specification (ACS) runtime to load a manifest and
evaluate intervention points in the application that hosts the OpenShell
sandbox.

```bash
pip install agent-control-specification
```

```python
from agent_control_specification import AgentControl

control = AgentControl.from_path("policies/agt-manifest.yaml")
```

See the [Agent Control Specification package guide](../packages/agent-control-specification.md)
for the current runtime and policy-authoring guidance.

## Migrating from the retired governance skill

### Option A: Governance Skill Inside the Sandbox (Python Library)

The pre-v5 `openshell-agentmesh` package is retained only for compatibility.
Importing `openshell_agentmesh` emits a `DeprecationWarning`; the
`GovernanceSkill`, `ShellPolicyViolation`, and `governed_shell` APIs are no
longer available. Replace that integration with an ACS manifest and host-level
intervention-point evaluation.

There is no OpenShell-specific replacement package. The ACS runtime is the
supported path for applying governance to an agent hosted in an OpenShell
sandbox.

## Related guidance

- [V5 removal and migration guidance](../v4-removal.md)
- [Agent Control Specification package guide](../packages/agent-control-specification.md)
