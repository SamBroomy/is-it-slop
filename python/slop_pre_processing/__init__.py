"""Docstring for slop_pre_processing."""

# Import only user-facing wrapper classes
from ._internal import TfidfVectorizer, VectorizerParams, __version__

__all__ = ["TfidfVectorizer", "VectorizerParams", "__version__"]
