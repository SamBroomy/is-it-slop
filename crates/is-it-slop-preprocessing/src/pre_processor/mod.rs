#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod vectorizer;

pub use vectorizer::{TfidfVectorizer, VectorizerParams};
