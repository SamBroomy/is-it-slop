# Type stubs for the extension module (crates/is-it-slop-preprocessing/src/python.rs).
from collections.abc import Iterable

__version__: str

def clean_for_training(iterable: Iterable[str]) -> list[str]: ...
def count_tokens(iterable: Iterable[str]) -> list[int]: ...
