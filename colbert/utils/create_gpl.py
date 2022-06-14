import gpl
import torch

DEVICE = torch.device("cuda")

base = "/workspace/jongyoon/datasets"

dataset = 'fiqa'
gpl.train(
    path_to_generated_data=f"{base}/generated/{dataset}",
    base_ckpt="distilbert-base-uncased",
    # base_ckpt='GPL/msmarco-distilbert-margin-mse',
    # The starting checkpoint of the experiments in the paper
    gpl_score_function="dot",
    # Note that GPL uses MarginMSE loss, which works with dot-product
    batch_size_gpl=32,
    gpl_steps=140000,
    new_size=-1,
    queries_per_passage=-1,

    output_dir=f"{base}/output/{dataset}",
    evaluation_data=f"{base}/{dataset}",
    evaluation_output=f"{base}/evaluation/{dataset}",
    generator="BeIR/query-gen-msmarco-t5-base-v1",
    retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
    retriever_score_functions=["cos_sim", "cos_sim"],
    # Note that these two retriever model work with cosine-similarity
    cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
    qgen_prefix="qgen",

    do_evaluation=True,
)
