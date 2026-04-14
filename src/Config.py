from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class ConfigGlobal:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\DeepLearning\D2L\Seq2Seq")
        data_dir = Path(r"E:\Dataset\D2L")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/DeepLearning/D2L/Seq2Seq")
        data_dir = Path("/media/fhz/Learning/Dataset/D2L")

    checkpoint_dir = wsf / "checkpoint"
    out_dir = wsf / "output"


cfgG = ConfigGlobal()
