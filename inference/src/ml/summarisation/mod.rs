use std::error::Error;
use std::fs;
use std::path::Path;
use std::result::Result;
use onnxruntime;
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::LoggingLevel;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray;
use onnxruntime::ndarray::Array;
use onnxruntime::ndarray::ArrayBase;
use onnxruntime::ndarray::Dim;
use onnxruntime::ndarray::OwnedRepr;
use onnxruntime::session::Session;
use tokenizers::Tokenizer;

use crate::ml::utils::tokenizer_utils::TokenizerUtils;
use crate::ml::{utils::model_utils::Config};

#[cfg_attr(feature = "summarisation_system_alloc", global_allocator)]
#[cfg(feature = "summarisation_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub fn main() -> Result<(), Box<dyn Error>> {
    let model_path_input = std::env::args()
        .skip(1)
        .next()
        .expect("Must provide a path to the model's directory");

    let model_path = Path::new(&model_path_input);

    let path_text = std::env::args()
        .skip(2)
        .next()
        .expect("Must provide a .txt file as the second arg");

    let text_content: String = fs::read_to_string(path_text)?.parse()?;

    let tokenizer = Tokenizer::from_path(model_path);

    // Scary "Ġ" character is normal: https://github.com/huggingface/transformers/issues/4786
    let encoding = tokenizer?.encode(text_content, true).unwrap();

    println!("{:?}", encoding.get_tokens());

    let ids = encoding
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();
    let id_array = Array::from_shape_vec((1, ids.len()), ids).unwrap();

    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    let mask_array = Array::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap();
    
    let embedding_array: Vec<ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>> = vec![id_array, mask_array];

    let environment = Environment::builder()
        .with_name("app")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        // .with_number_threads(4)?
        .with_model_from_file(model_path.join("model.onnx"))?;

    let config = Config::from_file(&model_path.join("config.json"))?;

    // NOTE: if not providing all inputs may be best to just use the encoder model variant?
    // @TODO: Get my inputs in the shape it wants
    // let prediction: _ = session.run(embedding_array.clone());

    println!("after");

    // # seem to use huggingface tokenizers lib with their own ONNX model consumed into onnxruntime
    // https://github.com/HIT-SCIR/libltp/blob/56689f6be39aa30350ea5755a804df19e461222a/ltp-rs/src/interface.rs
    // https://crates.io/crates/tokenizers


    Ok(())
}

pub type EmbeddingArray =
    Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>>;

fn print_inputs_outputs (session: Session) {
    println!("Inputs:");
    for (index, input) in session.inputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, input.name, input.input_type, input.dimensions
        )
    }

    println!("Outputs:");
    for (index, output) in session.outputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, output.name, output.output_type, output.dimensions
        );
    }
}
