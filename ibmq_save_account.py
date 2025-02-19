"""
Helper script to save a IBM Quantum account from data from a JSON file that
should look like this:

```json
{
  "channel": "ibm_quantum",
  "token": "...",
  "instance": "..."
}
```

and be located in `secret/ibmq_token.json`.
"""

import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN_PATH = Path("secret/ibmq_token.json")

if __name__ == "__main__":
    with TOKEN_PATH.open("r", encoding="utf8") as fp:
        kwargs = json.load(fp)
    QiskitRuntimeService.save_account(
        set_as_default=True, overwrite=True, **kwargs
    )
