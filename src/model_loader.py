#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib import import_module

from .jscc import JSCCAutoEncoder


_BACKBONES = {
    "cbjscc": (".backbones.cbjscc", "CBJSCCEnCoder", "CBJSCCDeCoder"),
    "adjscc": (".backbones.adjscc", "ADJSCCEnCoder", "ADJSCCDeCoder"),
    "deepjscc": (".backbones.deepjscc", "DeepJSCCEnCoder", "DeepJSCCDeCoder"),
}


def available_backbones():
    return sorted(_BACKBONES)


def select_backbone(backbone_name: str):
    try:
        module_name, encoder_name, decoder_name = _BACKBONES[backbone_name]
    except KeyError as exc:
        available = ", ".join(available_backbones())
        raise ValueError(
            f"Backbone {backbone_name} not found. Available: {available}"
        ) from exc

    module = import_module(module_name, package=__package__)
    return getattr(module, encoder_name), getattr(module, decoder_name)


def get_model(cfg):
    encoder_cls, decoder_cls = select_backbone(backbone_name=cfg.coder.name)
    return JSCCAutoEncoder(
        cfg=cfg,
        encoder=encoder_cls(cfg=cfg),
        decoder=decoder_cls(cfg=cfg),
    )
