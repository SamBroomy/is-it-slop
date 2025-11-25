"""TODO: Documentation for slop_pre_processing package."""

# Import only user-facing wrapper classes
from ._internal import TfidfVectorizer, VectorizerParams, __version__

# TODO: based on our internal design, is there any point in even having / exposing the vectorizer Params class?
__all__ = ["TfidfVectorizer", "VectorizerParams", "__version__"]
