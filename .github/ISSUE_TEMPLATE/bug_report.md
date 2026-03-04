---
name: Bug report
about: Something isn't working as expected
title: "[Bug] "
labels: bug
assignees: ""
---

## Describe the bug

_A clear, concise description of what went wrong._

## Steps to reproduce

```bash
# Minimal command or code snippet that reproduces the bug
testdata-ai generate --context ecommerce_customer --count 5
```

or

```python
from testdata_ai import generate
records = generate("ecommerce_customer", count=5)
```

## Expected behavior

_What did you expect to happen?_

## Actual behavior

_What happened instead? Paste error output / traceback here._

```
<error output>
```

## Environment

| | |
|---|---|
| testdata-ai version | _`pip show testdata-ai \| grep Version`_ |
| Python version | _`python --version`_ |
| OS | _e.g. macOS 14, Ubuntu 22.04_ |
| AI provider | _openai / anthropic / ollama_ |
| Provider model | _e.g. gpt-4o-mini_ |

## Additional context

_Any other details: .env config (without keys), custom contexts, etc._
