# VITATECS

VITATECS is a diagnostic VIdeo-Text dAtaset for the evaluation of TEmporal Concept underStanding. 

VITATECS is also available on [Huggingface](https://huggingface.co/datasets/lscpku/VITATECS). 

## Data

This repo contains 6 jsonl files under `data` folder, each of which corresponds to an aspect of temporal concepts (Direction, Intensity, Sequence, Localization, Compositionality, Type). 

Each line of the jsonl file is a json object, which contains the following fields:
- src_dataset: the name of the source dataset (VATEX or MSRVTT)
- video_name: the name of the video in the source dataset
- caption: the original caption of the video
- counterfactual: the generated counterfactual description of the video
- aspect: the aspect of temporal concepts that is modified

Example (indented for better presentation):
```
{
    "src_dataset": "VATEX", 
    "video_name": "i0ccSYMl0vo_000027_000037.mp4", 
    "caption": "A woman is placing a waxing strip on a man's leg.", 
    "counterfactual": "A woman is removing a waxing strip from a man's leg.",
    "aspect": "Direction"
}
```

## Evaluation

### Data Preparation

- Download the test set videos of MSRVTT and VATEX.
    - MSRVTT videos can be found [here](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt).
    - VATEX videos can be downloaded by following the instructions [here](https://eric-xw.github.io/vatex-website/download.html).
- Put the videos in the `videos` folder as follows:

```
videos
    MSRVTT
        video0.mp4
        video1.mp4
        ...
    VATEX
        _0ZBlXUcaOk_000013_000023.mp4
        _1qp63Hh6Xk_000015_000025.mp4
        ...
```

### ALPRO

The evaluation of ALPRO is implemented based on the [LAVIS](https://github.com/salesforce/LAVIS) library.

To evaluate ALPRO on VITATECS, run the following commands:

```bash
cd alpro
python eval_alpro.py alpro_pretrain.yaml
```

### X-CLIP/CLIP4Clip

The evaluation of X-CLIP/CLIP4Clip is implemented based on the [X-CLIP](https://github.com/xuguohai/X-CLIP) repository.

To evaluate X-CLIP/CLIP4Clip on VITATECS：
- Download ViT-B/32 and fine-tune the X-CLIP/CLIP4Clip model according to the instructions in [X-CLIP](https://github.com/xuguohai/X-CLIP).
- Change the `output_dir` and `init_model` arguments in `eval_vitatecs.sh`.
- Run `bash eval_vitatecs.sh`. 

## License

This dataset is under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
