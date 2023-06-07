# VITATECS

VITATECS is a diagnostic VIdeo-Text dAtaset for the evaluation of TEmporal Concept underStanding. 

## Data

This repo contains 6 jsonl files, each of which corresponds to an aspect of temporal concepts (Direction, Intensity, Sequence, Localization, Compositionality, Type). 

Each line of the jsonl file is a json object, which contains the following fields:
- src_dataset: the name of the source dataset (VATEX or MSRVTT)
- video_name: the name of the video in the source dataset
- caption: the original caption of the video
- counterfactual: the generated counterfactual description of the video

Example (indented for better presentation):
```
{
    "src_dataset": "VATEX", 
    "video_name": "i0ccSYMl0vo_000027_000037.mp4", 
    "caption": "A woman is placing a waxing strip on a man's leg.", 
    "counterfactual": "A woman is removing a waxing strip from a man's leg."
}
```

## License

This dataset is under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.