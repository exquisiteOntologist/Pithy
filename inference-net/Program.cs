using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// from debug dir: ./inference-net ./onnx/encoder_model.onnx ./onnx/sample-text.txt
string modelFilePath = args[0];
string articleTextFilePath = args[1];

// Read the image
Console.WriteLine ("Using model: " + modelFilePath);
Console.WriteLine ("Reading article: " + articleTextFilePath);

string articleTextContent = File.ReadAllText(articleTextFilePath);

Console.WriteLine($"Article Text: {articleTextContent}");

using InferenceSession session = new (modelFilePath);

session.InputMetadata.ToList().ForEach(key => Console.WriteLine($"In: {key}"));

// var tokenizer = can i just use any tokenizer?

Tensor<Int64> input_ids = new DenseTensor<Int64> (new [] { 0 });
Tensor<Int64> attention_mask = new DenseTensor<Int64> (new [] { 0 });

var inputs = new List<NamedOnnxValue> {
    NamedOnnxValue.CreateFromTensor("input_ids", input_ids),
    NamedOnnxValue.CreateFromTensor("attention_mask", attention_mask),
};

// session.Run(inputs);
