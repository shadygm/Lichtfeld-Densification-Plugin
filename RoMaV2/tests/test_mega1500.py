from romav2 import RoMaV2
from romav2.benchmarks import Mega1500
import logging
import wandb

wandb.init(project="roma-v2", name="mega1500")

logger = logging.getLogger(__name__)


def test_mega1500():
    model = RoMaV2()
    model.apply_setting("mega1500")
    mega1500 = Mega1500()
    res = mega1500.benchmark(model)
    wandb.log(res)
    logger.info(f"Mega1500 results: {res}")
    wandb.finish()


if __name__ == "__main__":
    test_mega1500()
