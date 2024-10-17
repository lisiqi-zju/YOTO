from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8Head

@MODELS.register_module()
class YOLOTOISTHead(YOLOv8Head):
    
