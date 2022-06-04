import os
import random
import time

import deepspeed
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta

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
    pe_sampling = 5
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      pe_sampling_size=pe_sampling,
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
    # ============= DEEPSPEED ============= #
    if args.deepspeed:
        "AssertionError: Amp and ZeRO are not currently compatible, " \
        "please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
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
                "enabled": True,
                # "loss_scale": 0,
                # "initial_scale_power": 32,
                # "loss_scale_window": 1000,
                # "hysteresis": 2,
                # "min_loss_scale": 1
            }
            # "amp": {
            #     "enabled": args.amp,
            # },
            # "zero_optimization": {
            #     "stage": 2,
            #     "offload_optimizer": {
            #         "device": "cpu",
            #         "pin_memory": True
            #     },
            #     "offload_param": {
            #         "device": "cpu",
            #         "pin_memory": True
            #     },
            #     "overlap_comm": True,
            #     "contiguous_gradients": True,
            #     "sub_group_size": 1e14,
            #     "reduce_bucket_size": "auto",
            #     "stage3_prefetch_bucket_size": "auto",
            #     "stage3_param_persistence_threshold": "auto",
            #     "stage3_max_live_parameters": 1e9,
            #     "stage3_max_reuse_distance": 1e9,
            #     "stage3_gather_fp16_weights_on_model_save": True
            # },
        }

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
                    print_progress(scores)

            model.step()

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
                    ans = labels.repeat(pe_sampling)
                    loss = criterion(scores, ans)
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

                print_message(f"{100 * batch_idx / len(list(batches))} % => ({batch_idx + 1} / {len(list(batches))})\n",
                              f"Average Loss: {avg_loss}")
                manage_checkpoints(args, colbert, optimizer, batch_idx + 1)
