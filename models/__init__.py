"""
模型定义模块
"""
# 基础模型
try:
    from .simple_cnn_multitask import SimpleCNNMultiTask
except ImportError:
    pass

try:
    from .ctranspath_multitask import get_ctranspath_multitask_model
except ImportError:
    pass

try:
    from .timm_multitask import get_timm_multitask_model
except ImportError:
    pass

# 旧版支持 (按需保留)
try:
    from .ctranspath import CTransPath, get_ctranspath_model
except ImportError:
    pass

try:
    from .ctranspath_legacy import get_ctranspath_legacy
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    get_ctranspath_legacy = None
