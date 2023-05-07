"""
Entrypoint of `llm_playground`.
"""
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import fire
from peft import LoraConfig
from transformers.trainer_utils import set_seed

from llm_playground.data import dataset_provider
from llm_playground.models import T5Trainer


def train_impl(
    _cls,
    pretrained_model_path: str,
    task_type: Literal["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"],
    dataset_name: Literal["STSB"],
    save_dir: str,
    lora_r: int = 8,
    lora_alpha: Optional[int] = None,
    lora_target_modules: Optional[Union[str, list[str]]] = None,
    lora_dropout: float = 0.0,
    pretrained_tokenizer_path: Optional[str] = None,
    load_in_8bit: bool = True,
    epochs: int = 3,
    batch_size: int = 32,
    micro_batch_size: int = 4,
    eval_batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_warmup_steps: int = 100,
    optimizer_name: Literal["adamw", "lamb", "lars", "lion", "sgd"] = "adamw",
    logging_steps: int = 100,
    eval_steps: int = 200,
    save_steps: int = 200,
    drop_last: bool = False,
    seed: int = 42,
):
    set_seed(seed)
    if pretrained_tokenizer_path is None:
        pretrained_tokenizer_path = pretrained_model_path
    if lora_alpha is None:
        lora_alpha = lora_r

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_to = str(Path(save_dir).resolve() / f"{start_time}")
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type,
    )

    trainer = T5Trainer(
        lora_cfg=lora_cfg,
        pretrained_model_path=pretrained_model_path,
        pretrained_tokenizer_path=pretrained_tokenizer_path,
        load_in_8bit=load_in_8bit,
    )

    dataset_fetcher, data_preprocessor, data_postprocessor, evaluator = dataset_provider(
        dataset_name, trainer.tokenizer, max_length=512
    )

    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps,
        optimizer_name=optimizer_name,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        evaluator=evaluator,
        dataset_fetcher=dataset_fetcher,
        data_preprocessor=data_preprocessor,
        data_postprocessor=data_postprocessor,
        save_to=save_to,
        drop_last=drop_last,
    )


def test_impl(
    _cls,
    load_from: str,
    dataset: str,
    seed: int = 42,
):
    set_seed(seed)
    print(f"This command is not implemented yet. You passed [{load_from}] and [{dataset}]")


def inference_impl(
    _cls,
    load_from: str,
    dataset: str,
    seed: int = 42,
):
    set_seed(seed)
    print(f"This command is not implemented yet. You passed [{load_from}] and [{dataset}]")


class launcher:
    """Entrypoint of LLM PlayGround. You can use `train`, `test`, `inference` commands."""

    train = train_impl
    test = test_impl
    inference = inference_impl


if __name__ == "__main__":
    fire.Fire(launcher)
