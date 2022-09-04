use std::error::Error;

use tokenizers::{Encoding};
use tokenizers::tokenizer::{Tokenizer};
use tokenizers::models::bpe::BPE;

// https://github.com/huggingface/tokenizers/tree/main/tokenizers
// BartTokenizer may not be available but Tokenizer.encode should do

// Use tokenizer defined in project dir instead of relating to existing model
// https://github.com/philschmid/rust-machine-learning/blob/bd714f033113ad55702ce2ead244e3e143f7558c/pipeline-rs/src/tokenizer_utils.rs
// https://github.com/philschmid/rust-machine-learning/blob/bd714f033113ad55702ce2ead244e3e143f7558c/pipeline-rs/src/modeling_utils.rs

pub fn main(string_for_encoding: String) -> Result<Encoding, Box<dyn Error>> {
    println!("Tokenizing:");
    println!("{}", string_for_encoding);

    let bpe_builder = BPE::from_file("./vocab.json", "./merges.txt");
    let bpe = match bpe_builder.build() {
        Ok(it) => it,
        Err(err) => return Err(err),
    };

    let tokenizer = Tokenizer::new(bpe);

    let encoding = match tokenizer.encode(string_for_encoding, false) {
        Ok(it) => it,
        Err(err) => return Err(err),
    };

    println!("{:?}", encoding.get_tokens());

    println!("After Encoding");

    Ok(encoding)
}
