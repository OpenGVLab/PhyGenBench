VisionEncoders = dict()


TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/MTScore/configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="bert-large-uncased",
    config="/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/MTScore/configs/config_bert_large.json",
    d_model=1024,
    fusion_layer=19,
)
TextEncoders["med_bert"] = dict(
    name="med_bert_base",
    pretrained="bert-base-uncased",
    config="/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/MTScore/configs/med_config.json",
    d_model=768,
)

TextEncoders["med_bert_large"] = dict(
    name="med_bert_large",
    pretrained="bert-base-uncased", # not a bug, it just follows BLIP.
    config="/mnt/petrelfs/mengfanqing/codebase_ljq/Phy_Score/MTScore/configs/med_large_config.json",
    d_model=768
)