# Type stubs for the compiled extension module (crates/is-it-slop-preprocessing/src/python/mod.rs).
from collections.abc import Iterable

__version__: str

def clean_for_training(iterable: Iterable[str], *, strip_markdown: bool = True) -> list[str]: ...
def count_tokens(iterable: Iterable[str]) -> list[int]: ...
