# Column subset selection

from types import MappingProxyType
from typing import Literal, Type

from t_arp.matrix.css_modules import (
    ARP_module,
    ARP_params,
    CSS_module,
    CSS_params,
    LeverageScoresSampling_module,
    LeverageScoresSampling_params,
    UniformSampling_module,
)

CSS_METHOD_IS_RANDOM_MAP = MappingProxyType(
    {"arp": True, "leverage_scores": True, "uniform": True}
)

CSS_METHOD_PARAMS_MAP = MappingProxyType(
    {
        "arp": ARP_params,
        "leverage_scores": LeverageScoresSampling_params,
        "uniform": CSS_params,
    }
)


def CSS_module_factory(
    method: Literal["arp", "uniform", "leverage_scores"],
) -> Type[CSS_module]:
    if method == "uniform":
        return UniformSampling_module
    if method == "arp":
        return ARP_module
    if method == "leverage_scores":
        return LeverageScoresSampling_module

    raise ValueError(f"Unknown method: {method}")
