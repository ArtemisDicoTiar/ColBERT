import os
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores, p=True):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    if p:
        print(f"#>>>  pos_avg: {positive_avg}, neg_avg: {negative_avg}\t\t|\t\t diff_avg: {positive_avg - negative_avg}")
    return positive_avg, negative_avg, positive_avg - negative_avg


def manage_checkpoints(args, colbert, optimizer, batch_idx):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if not args.deepspeed:
        if batch_idx % 2000 == 0:
            name = os.path.join(path, "colbert.dnn")
            save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments, args.deepspeed)

        if batch_idx in SAVED_CHECKPOINTS:
            name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
            save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments, args.deepspeed)

    else:
        if batch_idx % 10000 == 0:
            name = os.path.join(path, "colbert.dnn")
            save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments, args.deepspeed)