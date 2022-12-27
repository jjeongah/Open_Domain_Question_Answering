import logging
import os
import sys
import argparse
import wandb
import pytz
import datetime

from mrc import MRC
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def main(args):

    config = OmegaConf.load(f"./config/{args.config}.yaml")
    # wandb 설정
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    run_id = f"{config.wandb.name}_{now_time}"
    wandb.init(
        entity=config.wandb.team,
        project=config.wandb.project,
        group=config.model.name,
        id=run_id,
        tags=config.wandb.tags,
    )

    config.train.update(config.optimizer)
    if config.train.output_dir is None:
        config.train.output_dir = os.path.join("saved_models", config.model.name, run_id)
    training_args = TrainingArguments(**config.train)
    training_args.report_to = ["wandb"]

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("config", config)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(config.utils.seed)

    datasets = load_from_disk(config.path.train)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        from_tf=bool(".ckpt" in config.model.name),
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model.name,
        from_tf=bool(".ckpt" in config.model.name),
    )

    reader = MRC(
        config,
        training_args,
        tokenizer,
        model,
        datasets["train"],
        datasets["validation"],
    )
    reader.train()
    reader.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, _ = parser.parse_known_args()
    main(args)