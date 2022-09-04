use anyhow::{bail, Result};
use std::path::Path;

use tokenizers::Tokenizer;

// Original: https://github.com/philschmid/rust-machine-learning/blob/bd714f033113ad55702ce2ead244e3e143f7558c/pipeline-rs/src/tokenizer_utils.rs

pub trait TokenizerUtils {
    fn from_path(path: &Path) -> Result<Tokenizer> {
        match Tokenizer::from_file(path.join("tokenizer.json")) {
            Ok(tk) => Ok(tk),
            Err(err) => bail!("{}", err),
        }
    }
}

impl TokenizerUtils for Tokenizer {}
