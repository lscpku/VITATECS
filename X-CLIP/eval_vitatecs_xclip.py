from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from metrics import (
    compute_metrics,
    tensor_text_to_video_metrics,
    tensor_video_to_text_sim,
)
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_xclip import XCLIP
from modules.optimization import BertAdam

from util import parallel_apply, get_logger

from dataloaders.dataloader_vitatecs import VITATECS_DataLoader
from torch.utils.data import DataLoader

from tqdm import tqdm

global logger


def get_args(description="XCLIP on VITATECS"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--do_pretrain", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument("--train_csv", type=str, default="data/.train.csv", help="")
    parser.add_argument("--val_csv", type=str, default="data/.val.csv", help="")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/caption.pickle",
        help="data pickle file path",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/videos_feature.pickle",
        help="feature path",
    )

    parser.add_argument("--num_thread_reader", type=int, default=1, help="")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--batch_size_val", type=int, default=3500, help="batch size eval"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay"
    )
    parser.add_argument(
        "--n_display", type=int, default=100, help="Information display frequence"
    )
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="video feature dimension"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--max_frames", type=int, default=100, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument(
        "--hard_negative_rate",
        type=float,
        default=0.5,
        help="rate of intra negative sample",
    )
    parser.add_argument(
        "--negative_weighting",
        type=int,
        default=1,
        help="Weight the loss for intra negative",
    )
    parser.add_argument(
        "--n_pair", type=int, default=1, help="Num of pair to output from data loader"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cross_model",
        default="cross-base",
        type=str,
        required=False,
        help="Cross module",
    )
    parser.add_argument(
        "--init_model", default=None, type=str, required=False, help="Initial model."
    )
    parser.add_argument(
        "--resume_model",
        default=None,
        type=str,
        required=False,
        help="Resume train model.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="Changed in the execute process."
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--task_type",
        default="retrieval",
        type=str,
        help="Point the task `retrieval` to finetune.",
    )
    parser.add_argument(
        "--datatype", default="msrvtt", type=str, help="Point the dataset to finetune."
    )

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument(
        "--coef_lr", type=float, default=1.0, help="coefficient for bert branch."
    )
    parser.add_argument(
        "--use_mil",
        action="store_true",
        help="Whether use MIL as Miech et. al. (2020).",
    )
    parser.add_argument(
        "--sampled_use_mil",
        action="store_true",
        help="Whether MIL, has a high priority than use_mil.",
    )

    parser.add_argument(
        "--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text."
    )
    parser.add_argument(
        "--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual."
    )
    parser.add_argument(
        "--cross_num_hidden_layers", type=int, default=4, help="Layer NO. of cross."
    )

    parser.add_argument(
        "--loose_type",
        action="store_true",
        help="Default using tight type for retrieval.",
    )
    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")

    parser.add_argument(
        "--train_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )
    parser.add_argument(
        "--eval_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )

    parser.add_argument(
        "--freeze_layer_num",
        type=int,
        default=0,
        help="Layer NO. of CLIP need to freeze.",
    )
    parser.add_argument(
        "--slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.",
    )
    parser.add_argument(
        "--linear_patch",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="linear projection of flattened patches.",
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="meanP",
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="choice a similarity header.",
    )

    parser.add_argument(
        "--pretrained_clip_name",
        default="ViT-B/32",
        type=str,
        help="Choose a CLIP version",
    )

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_model(args, device):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location="cpu")
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
    )
    model = XCLIP.from_pretrained(
        args.cross_model,
        cache_dir=cache_dir,
        state_dict=model_state_dict,
        task_config=args,
    )

    model.to(device)

    return model


def eval_epoch(args, model, test_dataloader, device, aspect):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            (
                caption_input_ids,
                caption_input_mask,
                caption_segment_ids,
                counterfactual_input_ids,
                counterfactual_input_mask,
                counterfactual_segment_ids,
                video,
                video_mask,
            ) = batch

            B = video.size(0)

            (
                caption_sequence_output,
                caption_seq_features,
            ), caption_visual_output = model.get_sequence_visual_output(
                caption_input_ids,
                caption_segment_ids,
                caption_input_mask,
                video,
                video_mask,
            )
            (
                counterfactual_sequence_output,
                counterfactual_seq_features,
            ), counterfactual_visual_output = model.get_sequence_visual_output(
                counterfactual_input_ids,
                counterfactual_segment_ids,
                counterfactual_input_mask,
                video,
                video_mask,
            )

            caption_score = torch.cat(
                [
                    model.get_similarity_logits(
                        caption_sequence_output[i : i + 1],
                        caption_seq_features[i : i + 1],
                        caption_visual_output[i : i + 1],
                        caption_input_mask[i : i + 1],
                        video_mask[i : i + 1],
                        loose_type=model.loose_type,
                    )[0]
                    for i in range(B)
                ],
                dim=0,
            ).squeeze(1)
            counterfactual_score = torch.cat(
                [
                    model.get_similarity_logits(
                        counterfactual_sequence_output[i : i + 1],
                        counterfactual_seq_features[i : i + 1],
                        counterfactual_visual_output[i : i + 1],
                        counterfactual_input_mask[i : i + 1],
                        video_mask[i : i + 1],
                        loose_type=model.loose_type,
                    )[0]
                    for i in range(B)
                ],
                dim=0,
            ).squeeze(1)

            total += B
            correct += (caption_score > counterfactual_score).sum().item()

    acc = correct / total

    logger.info(f"{aspect}: {correct}/{total}={acc}")


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ClipTokenizer()

    model = init_model(args, device)

    ## ####################################
    # dataloader loading
    ## ####################################

    for aspect in [
        "Direction",
        "Type",
        "Intensity",
        "Localization",
        "Compositionality",
        "Sequence",
    ]:
        testset = VITATECS_DataLoader(
            vid_path="../videos",
            txt_path="../data",
            aspect=aspect,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            frame_order=args.eval_frame_order,
            slice_framepos=args.slice_framepos,
        )
        test_length = len(testset)

        test_loader = DataLoader(
            testset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
        )

        logger.info(f"***** Running test {aspect} *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_loader))

        eval_epoch(args, model, test_loader, device, aspect)


if __name__ == "__main__":
    main()
