"""
Some utils while building models
"""
import collections.abc
import logging
import os
import urllib.request
from itertools import repeat
from typing import List, Optional
from mindspore import load_checkpoint, load_param_into_net

def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None):
    """load pretrained model depending on cfgs of model"""
    if "url" not in default_cfg or not default_cfg["url"]:
        logging.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = "./pretain_ckpt"
    os.makedirs(download_path, exist_ok=True)

    filename = default_cfg["url"].split('/')[-1]
    filepath = download_path + '/' + filename
    if os.path.exists(filepath):
        print("File have already exist. skip")
    else:
        try:
            urllib.request.urlretrieve(default_cfg["url"], filename=filepath)
        except Exception as e:
            print("Error occurred when downloading file, error message:")
            print(e)

    param_dict = load_checkpoint(filepath)

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        logging.info("Converting first conv (%s) from 3 to 1 channel", conv1_name)
        con1_weight = param_dict[conv1_name + ".weight"]
        con1_weight.set_data(con1_weight.sum(axis=1, keepdims=True), slice_shape=True)
    elif in_channels != 3:
        raise ValueError("Invalid in_channels for pretrained weights")

    classifier_name = default_cfg["classifier"]
    if num_classes == 1000 and default_cfg["num_classes"] == 1001:
        classifier_weight = param_dict[classifier_name + ".weight"]
        classifier_weight.set_data(classifier_weight[:1000], slice_shape=True)
        classifier_bias = param_dict[classifier_name + ".bias"]
        classifier_bias.set_data(classifier_bias[:1000], slice_shape=True)
    elif num_classes != default_cfg["num_classes"]:
        params_names = list(param_dict.keys())
        param_dict.pop(
            _search_param_name(params_names, classifier_name + ".weight"),
            "No Parameter {} in ParamDict".format(classifier_name + ".weight"),
        )
        param_dict.pop(
            _search_param_name(params_names, classifier_name + ".bias"),
            "No Parameter {} in ParamDict".format(classifier_name + ".bias"),
        )

    if filter_fn is not None:
        param_dict = filter_fn(param_dict)

    load_param_into_net(model, param_dict)


def make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None,
) -> int:
    """Find the smallest integer larger than v and divisible by divisor."""
    if not min_value:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _search_param_name(params_names: List, param_name: str) -> str:
    for pi in params_names:
        if param_name in pi:
            return pi
    return ""