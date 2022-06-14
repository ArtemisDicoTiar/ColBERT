import os
import random
import time

import deepspeed
import string
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta

import wandb
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import AdamW, AutoModel

from colbert.modeling.kd_emb_colbert import KDEmbedColBERT
from colbert.utils.amp import MixedPrecisionManager

from colbert.distillation.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.utils.utils import print_message
from colbert.distillation.utils import print_progress, manage_checkpoints
from colbert.ds_utils.model import get_ds_model, get_ds_ce_model


def get_cross_embeds(model, reader, tokens, tok_range: dict):
    def _mask(input_ids):
        tokenizer = reader.cross_tokenizer.tok
        skiplist = {w: True
                    for symbol in string.punctuation
                    for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    tokens = list(map(lambda tok: tok.to(DEVICE), tokens))
    with torch.no_grad():
        tok_id, tok_mask = tokens
        cross_embed = model(*tokens).last_hidden_state

        cross_q_embeds = cross_embed[:, tok_range['Q'], :]
        cross_q_embeds = torch.nn.functional.normalize(cross_q_embeds, p=2, dim=2)

        cross_d_embeds = cross_embed[:, tok_range['D'], :]
        cross_d_mask = torch.tensor(_mask(tok_id[:, tok_range['D']]), device=DEVICE).unsqueeze(2).float()

        cross_d_embeds *= cross_d_mask
        cross_d_embeds = torch.nn.functional.normalize(cross_d_embeds, p=2, dim=2)
        return cross_q_embeds, cross_d_embeds


def distil(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    cross_encoder = AutoModel.from_pretrained("cross-encoder/ms-marco-electra-base")
    colbert = KDEmbedColBERT.from_pretrained('bert-base-uncased',
                                             pe_sampling_size=args.pe_sampling_size,
                                             query_maxlen=args.query_maxlen,
                                             doc_maxlen=args.doc_maxlen,
                                             dim=args.dim,
                                             similarity_metric=args.similarity,
                                             mask_punctuation=args.mask_punctuation)
    distill_criterion = MSELoss()  # soft (embed)
    student_criterion = CrossEntropyLoss()  # hard (score)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank < 1:
        wandb.init(project="pe-colbert", entity="artemisdicotiar")
        wandb.config.update(args)
    # ============= DEEPSPEED ============= #
    if args.deepspeed:
        colbert, optimizer = get_ds_model(args, colbert)
        cross_encoder = get_ds_ce_model(args, cross_encoder)

        labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

        start_time = time.time()
        train_loss = 0.0

        start_batch_idx = 0

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        batches = zip(range(start_batch_idx, args.maxsteps), reader)
        for batch_idx, BatchSteps in batches:
            this_batch_loss = 0.0

            ps, ns, ds = [], [], []
            for queries, passages, qd_pairs, col_range, cross_range in BatchSteps:
                cross_q_embeds, cross_d_embeds = get_cross_embeds(cross_encoder, reader, qd_pairs, cross_range)

                (hidden_Q, hidden_D), scores = colbert(queries, passages)
                col_q_embeds = hidden_Q[:, col_range['Q'], :]
                col_d_embeds = hidden_D[:, col_range['D'], :]
                scores = scores.view(2, -1).permute(1, 0)

                distill_loss_Q = distill_criterion(col_q_embeds, cross_q_embeds)
                distill_loss_D = distill_criterion(col_d_embeds, cross_d_embeds)

                distill_loss = (distill_loss_Q + distill_loss_D) / (2 * args.accumsteps)
                student_loss = student_criterion(scores, labels[:scores.size(0)]) / args.accumsteps

                # https://intellabs.github.io/distiller/knowledge_distillation.html
                # https://arxiv.org/pdf/1503.02531.pdf
                alpha = 0.5
                beta = 1 - alpha
                total_loss = alpha * distill_loss + beta * student_loss

                if args.rank < 1:
                    p, n, d = print_progress(scores, p=False)
                    ps.append(p)
                    ns.append(n)
                    ds.append(d)

                colbert.backward(total_loss)

                train_loss += total_loss.item()
                this_batch_loss += total_loss.item()

            colbert.step()

            if args.rank < 1:
                avg_loss = train_loss / (batch_idx + 1)

                num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
                elapsed = float(time.time() - start_time)

                # log_to_mlflow = (batch_idx % 20 == 0)
                # Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx,
                #                log_to_mlflow=log_to_mlflow)

                wandb.log({
                    'train/avg_loss': avg_loss,
                    'train/batch_loss': this_batch_loss,
                    'train/examples': num_examples_seen,
                    'train/throughput': num_examples_seen / elapsed,
                    "pos_avg": sum(ps) / len(ps),
                    "neg_avg": sum(ns) / len(ns),
                    "diff_avg": sum(ds) / len(ds)
                })

                progress = 100 * (batch_idx + 1) / len(range(start_batch_idx, args.maxsteps))
                if batch_idx % 100 == 0:
                    print_message(f"elapsed: {timedelta(seconds=elapsed)}",
                                  f"\tprogress:{progress}% ({batch_idx + 1}/{len(range(start_batch_idx, args.maxsteps))})",
                                  f"\tloss: {avg_loss}")

            # deepspeed's checkpoint must be called from all nodes and gpus.
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#saving-training-checkpoints
            manage_checkpoints(args, colbert, optimizer, batch_idx + 1)

    # ============= Original ============= #
    else:
        print("<Distillation>")
        cross_encoder = cross_encoder.to(DEVICE)
        cross_encoder.eval()
        colbert = colbert.to(DEVICE)
        colbert.train()

        if args.distributed:
            cross_encoder = torch.nn.parallel.DistributedDataParallel(cross_encoder, device_ids=[args.rank],
                                                                      output_device=args.rank,
                                                                      find_unused_parameters=True)

            colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                                output_device=args.rank,
                                                                find_unused_parameters=True)

        optimizer = AdamW(filter(lambda param: param.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
        optimizer.zero_grad()

        amp = MixedPrecisionManager(args.amp)

        labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

        start_time = time.time()
        train_loss = 0.0

        start_batch_idx = 0

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        batches = zip(range(start_batch_idx, args.maxsteps), reader)
        for batch_idx, BatchSteps in batches:
            this_batch_loss = 0.0

            ps, ns, ds = [], [], []
            for queries, passages, qd_pairs, col_range, cross_range in BatchSteps:
                cross_q_embeds, cross_d_embeds = get_cross_embeds(cross_encoder, reader, qd_pairs, cross_range)
                with amp.context():
                    (hidden_Q, hidden_D), scores = colbert(queries, passages)
                    col_q_embeds = hidden_Q[:, col_range['Q'], :]
                    col_d_embeds = hidden_D[:, col_range['D'], :]
                    scores = scores.view(2, -1).permute(1, 0)

                    distill_loss_Q = distill_criterion(col_q_embeds, cross_q_embeds)
                    distill_loss_D = distill_criterion(col_d_embeds, cross_d_embeds)

                    distill_loss = (distill_loss_Q + distill_loss_D) / (2 * args.accumsteps)
                    student_loss = 0
                    # student_loss = student_criterion(scores, labels[:scores.size(0)]) / args.accumsteps

                    # https://intellabs.github.io/distiller/knowledge_distillation.html
                    # https://arxiv.org/pdf/1503.02531.pdf
                    # alpha = 0.75
                    alpha = 1
                    beta = 1 - alpha
                    total_loss = alpha * distill_loss + beta * student_loss

                if args.rank < 1:
                    p, n, d = print_progress(scores, p=False)
                    ps.append(p)
                    ns.append(n)
                    ds.append(d)

                amp.backward(total_loss)

                train_loss += total_loss.item()
                this_batch_loss += total_loss.item()

            amp.step(colbert, optimizer)

            if args.rank < 1:
                avg_loss = train_loss / (batch_idx + 1)

                num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
                elapsed = float(time.time() - start_time)

                # log_to_mlflow = (batch_idx % 20 == 0)
                # Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
                # Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx,
                #                log_to_mlflow=log_to_mlflow)

                wandb.log({
                    'train/avg_loss': avg_loss,
                    'train/batch_loss': this_batch_loss,
                    'train/examples': num_examples_seen,
                    'train/throughput': num_examples_seen / elapsed,
                    "pos_avg": sum(ps) / len(ps),
                    "neg_avg": sum(ns) / len(ns),
                    "diff_avg": sum(ds) / len(ds)
                })

                progress = 100 * (batch_idx + 1) / len(range(start_batch_idx, args.maxsteps))
                if batch_idx % 10 == 0:
                    print_message(f"elapsed: {timedelta(seconds=elapsed)}",
                                  f"\tprogress:{progress}% ({batch_idx + 1}/{len(range(start_batch_idx, args.maxsteps))})",
                                  f"\tloss: {avg_loss}")
                manage_checkpoints(args, colbert, optimizer, batch_idx + 1)

        print_message(f"start: {time.ctime(start_time)}, elapsed: {time.ctime(time.time() - start_time)}")
        if args.rank < 1:
            wandb.finish()
