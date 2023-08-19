import sys
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from lavis.models import load_preprocess
from omegaconf import OmegaConf
from lavis.common.registry import registry


class TemporalBenchmark(Dataset):
    def __init__(self, vid_path, txt_path, aspect, vis_processor, txt_processor):
        super().__init__()
        self.aspect = aspect

        self.vid_path = vid_path
        self.data = []
        with open(os.path.join(txt_path, aspect + ".jsonl")) as f:
            for line in f:
                self.data.append(json.loads(line))

        self.vis_processor = vis_processor
        self.txt_processor = txt_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.vis_processor(
            os.path.join(
                self.vid_path,
                self.data[idx]["src_dataset"],
                self.data[idx]["video_name"],
            )
        )
        return (
            video,
            self.txt_processor(self.data[idx]["caption"]),
            self.txt_processor(self.data[idx]["counterfactual"]),
        )


def load_model_and_preprocess(name, config_path, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.
    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)
    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.load(config_path)

    # load model
    model = model_cls.from_config(cfg.get("model"))

    if is_eval:
        model.eval()

    # load preprocess
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        print(
            f"""No default preprocess for model {name} ({config_path}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu":
        model = model.float()

    return model.to(device), vis_processors, txt_processors


if __name__ == "__main__":
    print("Loading model")
    device = torch.device("cuda")
    config_path = sys.argv[1]
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="alpro_retrieval", config_path=config_path, is_eval=True, device=device
    )

    for aspect in [
        "Direction",
        "Type",
        "Intensity",
        "Localization",
        "Compositionality",
        "Sequence",
    ]:
        print("Loading dataset", aspect)

        testset = TemporalBenchmark(
            vid_path="../videos",
            txt_path="../data",
            aspect=aspect,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
        testloader = DataLoader(testset, batch_size=32, shuffle=False)

        print("Evaluating", aspect)
        total = 0
        correct = 0
        with torch.no_grad():
            for video, caption, counterfactual in tqdm(testloader):
                B, N, C, W, H = video.size()
                video = video.to(device, non_blocking=True)

                text = caption + counterfactual

                text_input = model.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                text_output = model.text_encoder.forward_text(
                    text_input,
                    token_type_ids=torch.zeros(
                        text_input.input_ids.shape, dtype=torch.long, device=device
                    ),
                )
                text_feats = text_output.last_hidden_state
                text_atts = text_input.attention_mask

                video_feat = model.visual_encoder.forward_features(video)  # B, M, H
                video_feat = torch.cat([video_feat, video_feat], dim=0)  # 2 * B, M, H
                video_atts = torch.ones(video_feat.size()[:-1], dtype=torch.long).to(
                    device, non_blocking=True
                )

                attention_mask = torch.cat([text_atts, video_atts], dim=1)
                embedding_output = torch.cat([text_feats, video_feat], dim=1)

                output = model.text_encoder(
                    encoder_embeds=embedding_output,
                    attention_mask=attention_mask,
                    return_dict=True,
                    mode="fusion",
                )

                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                caption_score = score[:B]
                counterfactual_score = score[B:]

                total += B
                correct += (caption_score > counterfactual_score).sum().item()

        print(aspect, correct, total, correct / total)
