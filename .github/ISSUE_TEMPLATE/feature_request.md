---
name: Feature request
about: Suggest an idea or improvement
title: "[Feature] "
labels: enhancement
assignees: ""
---

## Problem / motivation

_What use case or pain point does this address?_
_e.g. "I need to generate data for X, but there's no built-in context for it."_

## Proposed solution

_Describe the feature you'd like. Include an API / CLI sketch if relevant._

Example — new built-in context:
```bash
testdata-ai generate --context crypto_wallet --count 20
```

Example — new provider:
```bash
AI_PROVIDER=groq testdata-ai generate --context banking_user --count 5
```

Example — new output format:
```bash
testdata-ai generate --context hr_employee --format sql --count 100
```

## Alternatives considered

_Have you tried a workaround? e.g. custom context via `register_context()` or `--context-file`?_

## Additional context

_Screenshots, related issues, links to docs, etc._
