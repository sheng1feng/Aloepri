# Qwen Paper-Consistent Deployment Design

> **For agentic workers:** This design supersedes the narrower “bridge equivalence” objective. The goal is not to perfectly imitate the buffered redesign artifact, but to construct a deployment artifact that is closer to the original paper’s deployment-adaptation logic.

## Goal

Build a Qwen deployment line that is closer to the AloePri paper’s intended deployment semantics:

- standard Transformer runtime graph
- standard-visible export surface
- paper-relevant attention / FFN / norm perturbation retained as much as possible

## Current constraint

The repository now has:

- a `buffered redesign line` with promising security behavior
- a `standard-visible bridge line` with correct export/load properties but poor equivalence

The next phase must stop optimizing “bridge looks like buffered redesign” as an end in itself.

## Target architecture

The target is a new paper-consistent deployment artifact that:

1. can be loaded by standard HF/vLLM/SGLang-style tooling;
2. uses standard `model.* / lm_head.*` keys;
3. preserves as much paper-relevant component expression as possible directly in exported weights;
4. minimizes reliance on hidden buffered-only state.

## Priority work items

1. define the minimum retained paper-expression set for export;
2. replace placeholder bridge components with export-visible approximations or exact materializations;
3. re-run correctness and security audits on that new line.
