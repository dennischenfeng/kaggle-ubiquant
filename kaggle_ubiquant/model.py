"""
ML model
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any
from xgboost import XGBRegressor


@dataclass
class ModelConfig: 
    """
    Data for model configuration, including hparams
    """
    model_cls: Callable = XGBRegressor
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    

def generate_model(model_config: ModelConfig):
    """
    Initialize a model, ready to train on data.
    """
    return model_config.model_cls(**model_config.model_kwargs)