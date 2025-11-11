from __future__ import annotations
import time

class SimpleLogger:
    def __init__(self):
        self.t0 = time.time()

    def log(self, **kw):
        dt = time.time() - self.t0
        items = " ".join(f"{k}={v}" for k,v in kw.items())
        print(f"[{dt:7.1f}s] {items}")
