"""Own disposable file previews for the lifetime of one pipeline run."""
from pathlib import Path
from tempfile import TemporaryDirectory


class TemporaryCloudPreviews:
    """File callbacks must consume their preview before returning.

    Final exports use the requested output path separately and are retained.
    Memory previews never allocate a temporary directory.
    """

    def __init__(self, output_path: str, enabled: bool):
        self._directory = TemporaryDirectory(prefix="lichtfeld-densification-") if enabled else None
        self.base = (str(Path(self._directory.name) / f"{Path(output_path).stem}_intermediate")
                     if self._directory is not None else None)

    def cleanup(self):
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
