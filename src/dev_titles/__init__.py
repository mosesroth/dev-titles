"""dev_titles — deterministic execution harness for the subtitle-localizer skill.

The skill (.claude/skills/subtitle-localizer) carries the judgment; this package
carries the deterministic steps: transcription adapters, glossary retrieval/
enforcement, CJK re-segmentation + kinsoku, per-language validation, QE scoring,
subtitle format I/O, and publishing.

Vendor-facing adapters (transcribe/, translate/bedrock, publish/) are stubs until
model/region/account details are pinned for your deployment. The deterministic core
(config, model, segment, validate, glossary, formats) is implemented.
"""

__version__ = "0.2.2"
