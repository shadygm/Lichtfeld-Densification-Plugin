from romav2 import RoMaV2
from romav2.benchmarks import ScanNet1500
import logging
import wandb

wandb.init(project="roma-v2", name="scannet1500")

logger = logging.getLogger(__name__)


def test_scannet1500():
    model = RoMaV2()
    model.apply_setting("scannet1500")
    scannet1500 = ScanNet1500()
    res = scannet1500.benchmark(model)
    wandb.log(res)
    logger.info(f"ScanNet1500 results: {res}")


if __name__ == "__main__":
    test_scannet1500()
