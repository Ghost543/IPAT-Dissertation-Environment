"""Bounded Windows replacement retries; single-writer EXP-05 artefacts."""
import os
import tempfile
import time
from pathlib import Path

IO_VERSION = '0.1.1'


def atomic_write(path, writer, *, attempts=10, initial_delay=0.1, max_delay=1.0):
    """Serialize once to a unique closed sibling, then replace without deleting target.

    writer receives an open binary file. Persistent replacement failure retains both
    the previous target and the complete temporary candidate. This is not a
    multi-writer coordination mechanism or a power-loss durability guarantee.
    """
    if attempts < 1 or initial_delay < 0 or max_delay < 0:
        raise ValueError('Positive attempt count and non-negative delays required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w+b', prefix=path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as handle:
        candidate = Path(handle.name)
        try:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as error:
            error.add_note(f'Incomplete temporary file retained for inspection: {candidate}')
            raise
    delay = initial_delay
    for attempt in range(attempts):
        try:
            os.replace(candidate, path)
            return
        except OSError as error:
            # ERROR_ACCESS_DENIED / SHARING_VIOLATION / LOCK_VIOLATION.
            retryable = getattr(error, 'winerror', None) in (5, 32, 33)
            if not retryable or attempt == attempts - 1:
                error.add_note(f'Replacement failed; previous target was not deleted. '
                               f'Complete candidate retained at {candidate}. '
                               f'Target: {path}; attempts used: {attempt + 1}.')
                raise
            time.sleep(delay)
            delay = min(max_delay, delay * 2)
