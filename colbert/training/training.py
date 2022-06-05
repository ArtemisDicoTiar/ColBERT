import os
import random
import time

import deepspeed
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta

import wandb
from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(args):
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

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      pe_sampling_size=args.pe_sampling_size,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()
        wandb.init(project="pe-colbert", entity="artemisdicotiar")
        wandb.config.update(args)
    # ============= DEEPSPEED ============= #
    if args.deepspeed:
        config = {
            "train_batch_size": args.bsize,
            "gradient_accumulation_steps": args.accumsteps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": False,
            },
            # "amp": {
            #     "enabled": args.amp,
            # },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "cpu_offload": True
            },
        }
        if args.rank < 1:
            wandb.config.update({"deepspeed_config": config})

        deepspeed.init_distributed()

        model, optimizer, _, _ = deepspeed.initialize(model=colbert,
                                                      config_params=config,
                                                      model_parameters=colbert.parameters())

        criterion = nn.CrossEntropyLoss()
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

            for queries, passages in BatchSteps:
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps

                train_loss += loss.item()
                this_batch_loss += loss.item()

                model.backward(loss)

                if args.rank < 1:
                    p, n, d = print_progress(scores, p=False)
                    wandb.log({
                        "pos_avg": p,
                        "neg_avg": n,
                        "diff_avg": d
                    })

            model.step()

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
                })

                progress = 100 * (batch_idx + 1) / len(range(start_batch_idx, args.maxsteps))
                if batch_idx % 10 == 0:
                    print_message(f"elapsed: {timedelta(seconds=elapsed)}",
                                  f"\tprogress:{progress}% ({batch_idx + 1}/{len(range(start_batch_idx, args.maxsteps))})",
                                  f"\tloss: {avg_loss}")

            # deepspeed's checkpoint must be called from all nodes and gpus.
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#saving-training-checkpoints
            manage_checkpoints(args, model, optimizer, batch_idx + 1)

    # ============= Original ============= #
    else:
        colbert = colbert.to(DEVICE)
        colbert.train()

        if args.distributed:
            colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                                output_device=args.rank,
                                                                find_unused_parameters=True)

        optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
        optimizer.zero_grad()

        amp = MixedPrecisionManager(args.amp)
        criterion = nn.CrossEntropyLoss()
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

            for queries, passages in BatchSteps:
                with amp.context():
                    scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                    loss = criterion(scores, labels[:scores.size(0)])
                    loss = loss / args.accumsteps

                if args.rank < 1:
                    print_progress(scores)

                amp.backward(loss)

                train_loss += loss.item()
                this_batch_loss += loss.item()

            amp.step(colbert, optimizer)

            if args.rank < 1:
                avg_loss = train_loss / (batch_idx + 1)

                num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
                elapsed = float(time.time() - start_time)

                log_to_mlflow = (batch_idx % 20 == 0)
                Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx,
                               log_to_mlflow=log_to_mlflow)

                progress = 100 * (batch_idx + 1) / len(range(start_batch_idx, args.maxsteps))
                if batch_idx % 10 == 0:
                    print_message(f"elapsed: {timedelta(seconds=elapsed)}",
                                  f"\tprogress:{progress}% ({batch_idx + 1}/{len(range(start_batch_idx, args.maxsteps))})",
                                  f"\tloss: {avg_loss}")
                manage_checkpoints(args, colbert, optimizer, batch_idx + 1)

        print_message(f"start: {time.ctime(start_time)}, elapsed: {time.ctime(time.time() - start_time)}")
        if args.rank < 1:
            wandb.finish()
