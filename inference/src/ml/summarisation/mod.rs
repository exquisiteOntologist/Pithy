use std::error::Error;
use std::fs;
use std::path::Path;
use std::result::Result;
use libc::int64_t;
use onnxruntime;
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::LoggingLevel;
use onnxruntime::TypeToTensorElementDataType;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray;
use onnxruntime::ndarray::Array;
use onnxruntime::ndarray::ArrayBase;
use onnxruntime::ndarray::Dim;
use onnxruntime::ndarray::OwnedRepr;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::tensor::ndarray_tensor;
use tokenizers::Tokenizer;

use crate::ml::utils::tokenizer_utils::TokenizerUtils;
use crate::ml::{utils::model_utils::Config};
use crate::summarisation::onnxruntime::TensorElementDataType;

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
    let id_shape = (1, ids.len());
    let id_array: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = ndarray::Array::from_shape_vec(id_shape, ids.clone()).unwrap();

    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    let mask_shape = (1, attention_mask.len());
    let mask_array = ndarray::Array::from_shape_vec(mask_shape, attention_mask.clone()).unwrap();

    // Vec<Array<TIn, D>>
    pub type EmbeddingArray =
        Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>>;

    // let embedding_array: EmbeddingArray = vec![id_array, mask_array];
    // let embedding_array: EmbeddingArray = vec![id_array];
    // let x = Tensor::from_array(embedding_array);
    // let embedding_array: Vec<ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>> = vec![id_array, mask_array];

    let environment = Environment::builder()
        .with_name("app")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        // .with_number_threads(4)?
        .with_model_from_file(model_path.join("model.onnx"))?;

    let config = Config::from_file(&model_path.join("config.json"))?;

    print_inputs_outputs(&session);

    /* let array = ndarray::Array::linspace(0.0_f32, 1.0, 100);
    // Multiple inputs and outputs are possible
    let input_tensor = vec![array];
    let outputs: Vec<OrtOwnedTensor<f32,_>> = session.run(input_tensor)?; */

    // let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
    // let output0_shape: Vec<usize> = session.outputs[0]
    //     .dimensions()
    //     .map(|d| d.unwrap())
    //     .collect();

    // let input1_shape: Vec<usize> = session.inputs[1].dimensions().map(|d| d.unwrap()).collect();
    // let output1_shape: Vec<usize> = session.outputs[1]
    //     .dimensions()
    //     .map(|d| d.unwrap())
    //     .collect();

    // assert_eq!(input0_shape, [1, 3, 224, 224]);
    // assert_eq!(output0_shape, [1, 1000, 1, 1]);

    // initialize input data with values in [0.0, 1.0]
    // let n: u32 = session.inputs[0]
    //     .dimensions
    //     .iter()
    //     .map(|d| d.unwrap())
    //     .product();
    // let array_0 = Array::linspace(0.0_f32, 1.0, n as usize)
    // let array_0 = Array::linspace(0.0_f32, 1.0, n as usize)
    let array_0: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = id_array;
    let array_1: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> = mask_array;
    let input_tensor_values = vec![array_0, array_1];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;


    // assert_eq!(outputs[0].shape(), id_shape.as_slice());
    for i in 0..5 {
        println!("Output 0: Score for class [{}] =  {}", i, outputs[0][[0, i, 0, 0]]);
    }

    // assert_eq!(outputs[1].shape(), mask_shape.as_slice());
    for i in 0..5 {
        println!("Output 1: Score for class [{}] =  {}", i, outputs[1][[0, i, 0, 0]]);
    }
   
   
    // TypeToTensorElementDataType::tensor_element_data_type(i64);
    // // TypeToTensorElementDataType
    // let x = Tensor::from_array(embedding_array);
    // let tensor_data_type = i64::tensor_element_data_type();
    // let = TensorElementDataType::Int64;

    // pub type EmbeddingArray =
    //     Vec<ndarray::ArrayBase<ndarray::OwnedRepr<tensor_data_type>, ndarray::Dim<[usize; 2]>>>;

    // pub type EmbeddingArray =
    //     Vec<ndarray::ArrayBase<ndarray::OwnedRepr<TensorElementDataType>, ndarray::Dim<[usize; 2]>>>;


    // NOTE: if not providing all inputs may be best to just use the encoder model variant?
    // @TODO: Get my inputs in the shape it wants
    // let prediction: _ = session.run(embedding_array.clone());

    println!("after");

    // # seem to use huggingface tokenizers lib with their own ONNX model consumed into onnxruntime
    // https://github.com/HIT-SCIR/libltp/blob/56689f6be39aa30350ea5755a804df19e461222a/ltp-rs/src/interface.rs
    // https://crates.io/crates/tokenizers


    Ok(())
}

// pub type EmbeddingArray =
//     Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>>;

// crate::onnxruntime::<Int64>

// TensorElementDataType<Int64>



// pub type EmbeddingArray =
//     Vec<ndarray::ArrayBase<ndarray::OwnedRepr<Int64>, ndarray::Dim<[usize; 2]>>>;

/* 
    Inputs:
    0:
        name = input_ids
        type = Int64
        dimensions = [None, None]
    1:
        name = attention_mask
        type = Int64
        dimensions = [None, None]
    2:
        name = decoder_input_ids
        type = Int64
        dimensions = [None, None]
    3:
        name = decoder_attention_mask
        type = Int64
        dimensions = [None, None]
    Outputs:
    0:
        name = logits
        type = Float
        dimensions = [None, None, Some(50264)]
    1:
        name = onnx::MatMul_2341
        type = Float
        dimensions = [None, None, Some(1024)]
*/
fn print_inputs_outputs (session: &Session) {
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
