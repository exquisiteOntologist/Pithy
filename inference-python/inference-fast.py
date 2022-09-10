from transformers import pipeline
from transformers import BartTokenizerFast
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers


# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
  
#   assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

#   # Few properties that might have an impact on performances (provided by MS)
#   options = SessionOptions()
#   options.intra_op_num_threads = 1
#   options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

#   # Load the model as a graph and prepare the CPU backend 
#   session = InferenceSession(model_path, options, providers=[provider])
#   session.disable_fallback()
    
#   return session


# @contextmanager
# def track_infer_time(buffer: [int]):
#     start = time()
#     yield
#     end = time()

#     buffer.append(end - start)


# @dataclass
# class OnnxInferenceResult:
#   model_inference_time: [int]  
#   optimized_model_path: str

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))


# https://huggingface.co/docs/transformers/model_doc/bart
# tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
# cpu_model = create_model_for_provider("onnx/encoder_model.onnx", "CPUExecutionProvider")

# # Inputs are provided through numpy array
# model_inputs = tokenizer(ARTICLE, return_tensors="pt")
# inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

# # Run the model (None = get all the outputs)
# sequence, pooled = cpu_model.run(None, inputs_onnx)

# # Print information about outputs

# print(f"Sequence output: {sequence.shape}, Pooled output: {pooled.shape}")




