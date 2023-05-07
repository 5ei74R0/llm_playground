import os
import shutil
from typing import Literal, Optional, Union

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from peft.import_utils import is_bnb_available
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from llm_playground.data import DatasetFetcher, Evaluator, PostProcessor, PreProcessor
from llm_playground.utils import reproducible_worker_init_fn


class T5Trainer:
    def __init__(
        self,
        lora_cfg: LoraConfig,
        pretrained_model_path: str = "google/flan-t5-xl",
        pretrained_tokenizer_path: str = "google/flan-t5-xl",
        load_in_8bit: bool = True,
    ) -> None:
        super().__init__()
        # Validate inputs
        if load_in_8bit and not is_bnb_available():
            raise RuntimeError("8-bit quantization is not available. Please install bitsandbytes.")

        self.load_in_8bit = load_in_8bit

        # Load tokenizer
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(pretrained_tokenizer_path)

        # Load model
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )
        if load_in_8bit:
            self.base_model = self._typing_wrapped_prepare_model_for_int8_training(self.base_model)
        self.model: Union[PreTrainedModel, PeftModel] = get_peft_model(self.base_model, lora_cfg)

    @torch.inference_mode()
    def _eval_loop(self, dataloader, postprocessor, evaluator: Evaluator):
        self.model.eval()
        for batch_idx, batch in enumerate(dataloader):
            batch.to(self.model.device)
            outputs: torch.Tensor = self.model.generate(inputs=batch.input_ids)
            decoded_outputs: list[str] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            evaluator.update(postprocessor(decoded_outputs, batch))
            if batch_idx == 0:
                print(f"Example output: {decoded_outputs}")
                print(
                    f"        labels: {self.tokenizer.batch_decode(torch.where(batch.labels < 0, 0, batch.labels), skip_special_tokens=True)}"
                )
                print(f"        answer: {batch.score.tolist()}")
        self.model.train()
        return evaluator.compute()

    def _estimate_total_steps(
        self, gradient_accumulation_steps: int, epochs: int, train_dataloader: DataLoader
    ) -> int:
        iterations_per_epoch = len(list(train_dataloader))
        steps_per_epoch = (iterations_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
        return epochs * steps_per_epoch

    def _get_optimizer(
        self, optimizer_name: Literal["adamw", "lamb", "lars", "lion", "sgd"], learning_rate: float = 1e-4
    ) -> bnb.optim.optimizer.Optimizer8bit:
        optim_bits: int = 32
        if self.load_in_8bit:
            optim_bits = 8
        if optimizer_name == "adamw":
            return bnb.optim.AdamW(self.model.parameters(), lr=learning_rate, optim_bits=optim_bits)
        elif optimizer_name == "lamb":
            return bnb.optim.LAMB(self.model.parameters(), lr=learning_rate, optim_bits=optim_bits)
        elif optimizer_name == "lars":
            return bnb.optim.LARS(self.model.parameters(), lr=learning_rate, optim_bits=optim_bits)
        elif optimizer_name == "lion":
            return bnb.optim.Lion(self.model.parameters(), lr=learning_rate, optim_bits=optim_bits)
        elif optimizer_name == "sgd":
            return bnb.optim.SGD(self.model.parameters(), lr=learning_rate, optim_bits=optim_bits)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _prepare_dataloaders(
        self,
        dataset_fetcher: DatasetFetcher,
        data_preprocessor: PreProcessor,
        micro_batch_size: int,
        eval_batch_size: int,
    ):
        train_split, eval_split, test_split = dataset_fetcher()
        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 0
        train_dataloader = DataLoader(
            train_split.map(data_preprocessor),
            micro_batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                padding=True,
                return_tensors="pt",
            ),
            pin_memory=True,
            num_workers=n_cpus,
            worker_init_fn=reproducible_worker_init_fn,
        )
        eval_dataloader = DataLoader(
            eval_split.map(data_preprocessor),
            eval_batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                padding=True,
                return_tensors="pt",
            ),
            pin_memory=True,
            num_workers=n_cpus,
            worker_init_fn=reproducible_worker_init_fn,
        )
        test_dataloader = DataLoader(
            test_split.map(data_preprocessor),
            eval_batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                padding=True,
                return_tensors="pt",
            ),
            pin_memory=True,
            num_workers=n_cpus,
            worker_init_fn=reproducible_worker_init_fn,
        )
        return train_dataloader, eval_dataloader, test_dataloader

    def _typing_wrapped_prepare_model_for_int8_training(
        self, model, use_gradient_checkpointing=True
    ) -> T5ForConditionalGeneration:
        return prepare_model_for_int8_training(model, use_gradient_checkpointing)

    def get_raw_model_ref(self) -> PreTrainedModel:
        return self.base_model

    def get_peft_model_ref(self) -> PeftModel:
        return self.model

    def get_tokenizer_ref(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def train(
        self,
        epochs: int,
        batch_size: int,
        micro_batch_size: int,
        eval_batch_size: int,
        learning_rate: float,
        num_warmup_steps: int,
        optimizer_name: Literal["adamw", "lamb", "lars", "lion", "sgd"],
        logging_steps: int,
        eval_steps: int,
        save_steps: int,
        # train_dataloader: DataLoader,
        # eval_dataloader: DataLoader,
        evaluator: Evaluator,
        dataset_fetcher: DatasetFetcher,
        data_preprocessor: PreProcessor,
        data_postprocessor: PostProcessor,
        save_to: str,
        drop_last: bool = False,
    ):
        # Validate inputs
        if batch_size % micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size.")
        # if train_dataloader.batch_size != micro_batch_size:
        #     raise ValueError("Train dataloader batch size must be equal to micro batch size.")
        # if eval_dataloader.batch_size != eval_batch_size:
        #     raise ValueError("Eval dataloader batch size must be equal to eval batch size.")

        # Prepare Dataloader
        train_dataloader, eval_dataloader, test_dataloader = self._prepare_dataloaders(
            dataset_fetcher, data_preprocessor, micro_batch_size, eval_batch_size
        )

        # Setup
        min_loss = float("inf")
        save_dir_name: Optional[str] = None
        global_step = 0
        gradient_accumulation_steps = batch_size // micro_batch_size
        estimated_total_steps = self._estimate_total_steps(gradient_accumulation_steps, epochs, train_dataloader)
        training_progress = tqdm(
            total=estimated_total_steps, desc="Training", position=0, leave=True, dynamic_ncols=True
        )
        optimizer = self._get_optimizer(optimizer_name, learning_rate)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=estimated_total_steps,
        )
        # Initial evaluation
        evaluator.reset()
        eval_result = self._eval_loop(eval_dataloader, data_postprocessor, evaluator)
        print(f"Epoch {0:02d}, Step {global_step:06d}, Eval: {eval_result}")
        for epoch_idx in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                batch.to(self.model.device)
                outputs: Seq2SeqLMOutput = self.model(batch.input_ids, labels=batch.labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if batch_idx % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    training_progress.update(1)
                    for param in self.model.parameters():  # fast zero_grad
                        param.grad = None

                    if global_step % logging_steps == 0:
                        print(f"Epoch {epoch_idx:02d}, Step {global_step:06d}, Loss: {loss.item()}")

                    if global_step % eval_steps == 0:
                        evaluator.reset()
                        eval_result = self._eval_loop(eval_dataloader, data_postprocessor, evaluator)
                        print(f"Epoch {epoch_idx:02d}, Step {global_step:06d}, Eval: {eval_result}")

                    if global_step % save_steps == 0:
                        self.model.save_pretrained(f"{save_to}-epoch-{epoch_idx:02d}-step-{global_step:06d}")

            if not drop_last and batch_idx % gradient_accumulation_steps != 0:
                outputs.loss.backward()
                optimizer.step()
                global_step += 1
                for param in self.model.parameters():  # fast zero_grad
                    param.grad = None

                if global_step % logging_steps == 0:
                    print(f"Epoch {epoch_idx:02d}, Step {global_step:06d}, Loss: {loss.item()}")

                if global_step % eval_steps == 0:
                    eval_result = self._eval_loop(eval_dataloader, data_postprocessor, evaluator)
                    print(f"Epoch {epoch_idx:02d}, Step {global_step:06d}, Eval: {eval_result}")

                if global_step % save_steps == 0:
                    if loss.item() < min_loss:
                        if save_dir_name is not None:
                            shutil.rmtree(save_dir_name)
                        self.model.save_pretrained(f"{save_to}-epoch-{epoch_idx:02d}-step-{global_step:06d}")
                        save_dir_name = f"{save_to}-epoch-{epoch_idx:02d}-step-{global_step:06d}"

        # Test
        evaluator.reset()
        test_result = self._eval_loop(test_dataloader, data_postprocessor, evaluator)
        print(f"Test: {test_result}")

    @torch.inference_mode()
    def test(
        self,
        micro_batch_size: int,
        eval_batch_size: int,
        evaluator: Evaluator,
        dataset_fetcher: DatasetFetcher,
        data_preprocessor: PreProcessor,
        data_postprocessor: PostProcessor,
    ):
        # Prepare Dataloader
        _, _, test_dataloader = self._prepare_dataloaders(
            dataset_fetcher, data_preprocessor, micro_batch_size, eval_batch_size
        )

        evaluator.reset()
        test_result = self._eval_loop(test_dataloader, data_postprocessor, evaluator)
        print(f"Test: {test_result}")
