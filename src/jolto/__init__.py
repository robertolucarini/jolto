import os
import tempfile
from pathlib import Path

# OpenMP duplicate-runtime workaround; harmless on all platforms
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# portable Inductor cache location; setdefault keeps a user override
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", str(Path(tempfile.gettempdir()) / "torchinductor_cache")
)
import torch

torch.set_num_threads(os.cpu_count() or 1)
torch.backends.cudnn.benchmark = True
__version__ = "0.1.0"
