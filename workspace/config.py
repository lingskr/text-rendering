import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
FONT_DIR = DATA_DIR / "font"
TEXT_DIR = DATA_DIR / "corpus"

# Font configuration
font_cfg = dict(
    font_dir=FONT_DIR,
    font_size=(30, 31),  # Adjust font size as needed
)

# Perspective transformation
perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)

# Corpus configuration for Myanmar text
def get_myanmar_corpus():
    return WordCorpus(
        WordCorpusCfg(
            text_paths=[TEXT_DIR / "burmse.txt"],  # Myanmar text file
            filter_by_chars=False,  # You can enable character filtering if needed
            length=(2, 5),  # Adjust length according to your need
            **font_cfg
        ),
    )

# Base configuration function
def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=50,  # Number of images to generate
        save_dir=OUT_DIR / name,  # Output directory
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,  # Background images directory
            perspective_transform=perspective_transform,
            gray=gray,  # Generate grayscale images if True
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )

# Myanmar text rendering configuration
def myanmar_data():
    return base_cfg(
        "myanmar_data",
        corpus=get_myanmar_corpus(),
        corpus_effects=Effects(
            [
                Line(0.5, color_cfg=FixedTextColorCfg()),  # Line effect with fixed color
                OneOf([DropoutRand(), DropoutVertical()]),  # Random dropout effect
            ]
        ),
        layout_effects=Effects(Line(p=1)),
    )

# Example: Emboss effect using imgaug for Myanmar text
def imgaug_emboss_example():
    return base_cfg(
        "imgaug_emboss_example",
        corpus=get_myanmar_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
            ]
        ),
    )

# The configuration file must have a configs variable
configs = [
    myanmar_data(),
    imgaug_emboss_example(),
]
