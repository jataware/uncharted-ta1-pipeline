import json
import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, Union, Iterable


class MobileNetRCNN(FasterRCNN):

    """
    Pytorch model definition. MobileNetV2 backbone with FasterRCNN head.
    """

    def __init__(self,
                 config: dict,
                 state_dict: Union[None, Dict[str, torch.Tensor]] = None,
                 ) -> None:
        # TODO: Make Pydantic Pytorch model config object.
        assert config['num_classes'] > 0 and isinstance(config['num_classes'], int)
        assert all(i > 0 for i in config['anchor_sizes']) and isinstance(config['anchor_sizes'], Iterable)
        assert all(i > 0 for i in config['aspect_ratios']) and isinstance(config['aspect_ratios'], Iterable)
        assert config['roi_output_size'] > 0 and isinstance(config['roi_output_size'], int)
        assert config['samping_ratio'] > 0 and isinstance(config['samping_ratio'], int)

        # Always load model from random initialization, or from our own pretrained checkpoint. Loading directly from third party hosted checkpoints is not supported.
        backbone = torchvision.models.mobilenet_v2().features
        backbone.out_channels = 1280

        anchor_generator = AnchorGenerator(sizes=(config['anchor_sizes'],),
                                           aspect_ratios=(config['aspect_ratios'],))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=config['roi_output_size'],
                                                        sampling_ratio=config['samping_ratio'])
        self.config = config
        super().__init__(backbone=backbone,
                         num_classes=config['num_classes'],
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler)

        self._load_pretrained_state_dict(state_dict)

    @classmethod
    def from_pretrained(cls, path: str):
        config = json.load(open(os.path.join(path + '/config.json'), 'r', encoding='utf-8'))
        state_dict = torch.load(os.path.join(path + '/state_dict.pt'), map_location=torch.device('cpu'))
        return cls(state_dict=state_dict, config=config)

    def _load_pretrained_state_dict(self,
                                    state_dict: Union[Dict[str, torch.tensor], None] = None) -> None:
        if state_dict:
            self.load_state_dict(state_dict, strict=True)
