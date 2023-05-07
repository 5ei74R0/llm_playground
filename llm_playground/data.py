from typing import Any, Literal, NamedTuple, Optional

from scipy.stats import pearsonr, spearmanr
from torch.utils.data import IterDataPipe
from torchtext.datasets import STSB
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


# # class BatchEncodingPlus(UserDict):
# #     def __init__(self, batch_encoding: BatchEncoding, additional_data: dict[str, Any], **kwargs) -> None:
# #         super().__init__(**kwargs)
# #         self.batch_encoding = batch_encoding
# #         self.additional_data: dict[str, Any] = additional_data
# class BatchEncodingPlus(BatchEncoding):
#     def __init__(
#         self,
#         data: "dict[str, Any] | None" = None,
#         encoding: "EncodingFast | Sequence[EncodingFast] | None" = None,
#         tensor_type: "str | TensorType | None" = None,
#         prepend_batch_axis: bool = False,
#         n_sequences: "int | None" = None,
#     ):
#         super().__init__(data, encoding, tensor_type, prepend_batch_axis, n_sequences)
#         self.additional_data: dict[str, Any] = {}

#     # def __init__(self, *args, **kwargs) -> None:
#     #     self.data: dict[str, Any] = {}
#     #     super().__init__(*args, **kwargs)
#     #     self.additional_data: dict[str, Any] = {}

#     def to(self, device: Union[str, torch.device]) -> "BatchEncodingPlus":
#         """
#         Send all values to device by calling `v.to(device)` (PyTorch only).

#         Args:
#             device (`str` or `torch.device`): The device to put the tensors on.

#         Returns:
#             [`BatchEncoding`]: The same instance after modification.
#         """
#         requires_backends(self, ["torch"])

#         # This check catches things like APEX blindly calling "to" on all inputs to a module
#         # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
#         # into a HalfTensor
#         if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
#             self.data = {
#                 k: v.to(device=device) for k, v in self.data.items() if isinstance(v, torch.Tensor)  # type: ignore
#             }  # not only tensors in values
#         else:
#             logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
#         return self


class DataCollatorForSeq2SeqPlus(DataCollatorForSeq2Seq):
    # def __init__(
    #     self,
    #     tokenizer: PreTrainedTokenizerBase,
    #     model: Optional[Any] = None,
    #     padding: Union[bool, str, PaddingStrategy] = True,
    #     max_length: Optional[int] = None,
    #     pad_to_multiple_of: Optional[int] = None,
    #     label_pad_token_id: int = -100,
    #     return_tensors: str = "pt",
    # ) -> None:
    #     self.internal_collate_fn = DataCollatorForSeq2Seq(
    #         tokenizer,
    #         model,
    #         padding,
    #         max_length,
    #         pad_to_multiple_of,
    #         label_pad_token_id,
    #         return_tensors,
    #     )
    def __call__(
        self, batch_encoding_plus: list[BatchEncoding], return_tensors=None, *args: Any, **kwds: Any
    ) -> BatchEncoding:
        additional_data_container = []
        for batch_enc in batch_encoding_plus:
            additional_data_container.append(batch_enc.pop("additional_data", default={}))
        batch_encoding = super().__call__(batch_encoding_plus, return_tensors=return_tensors, *args, **kwds)
        batch_encoding["additional_data"] = additional_data_container
        return batch_encoding

    # def __call__(self, batch_encoding_plus: BatchEncodingPlus, return_tensors=None) -> BatchEncoding:
    #     padded_batch_encoding = self.internal_collate_fn.__call__(batch_encoding_plus.batch_encoding, return_tensors)
    #     padded_batch_encoding["additional_data"] = batch_encoding_plus.additional_data
    #     return padded_batch_encoding


class DatasetFetcher:
    def __init__(self) -> None:
        pass

    def __call__(self, dataset: Optional[str] = None) -> tuple[IterDataPipe, IterDataPipe, IterDataPipe]:
        raise NotImplementedError()


class EvalInput(NamedTuple):
    predictions: Any
    labels: Any


class Evaluator:
    def __init__(self) -> None:
        pass

    def update(self, eval_input: EvalInput, **kwargs) -> None:
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class PreProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, data: Any) -> BatchEncoding:
        raise NotImplementedError()


class PostProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, decoded_prediction: list[str], batch: BatchEncoding) -> EvalInput:
        raise NotImplementedError()


class Seq2SeqDataBaggage(NamedTuple):
    prompt: str
    answer: str
    others: dict[str, Any]


def parse_s2f(s: str, default: int = -1) -> float:
    try:
        return float(s)
    except ValueError:
        return float(default)


class DatasetFetcherStsb(DatasetFetcher):
    def __call__(self, dataset: Optional[str] = None) -> tuple[IterDataPipe, IterDataPipe, IterDataPipe]:
        train, dev, test = STSB()
        return train, dev, test


# def load_stsb_dataset() -> tuple[IterDataPipe, IterDataPipe, IterDataPipe]:
#     train, dev, test = STSB()
#     return train, dev, test


# def preprocess_stsb(idx: int, score: float, sentence1: str, sentence2: str):
#     # Reformat into seq2seq format
#     prompt = f"stsb sentence1: {sentence1} sentence2: {sentence2}"
#     target = f"{round(score * 5) / 5:.1f}"
#     return Seq2SeqDataBaggage(
#         prompt,
#         target,
#         {
#             "idx": idx,
#             "score": score,
#         },
#     )


class PreProcessorStsb(PreProcessor):
    def __call__(self, data: tuple[int, int, str, str]) -> BatchEncoding:
        # STSB data format
        idx, score, sentence1, sentence2 = data

        # Reformat into seq2seq format
        prompt = f"stsb sentence1: {sentence1} sentence2: {sentence2}"
        answer = f"{round(score * 5) / 5:.1f}"

        batch_encoding: BatchEncoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
        )
        batch_encoding["labels"] = self.tokenizer.encode(answer, max_length=self.max_length, truncation=True)

        # return BatchEncodingPlus(
        #     batch_encoding,
        #     additional_data={
        #         "idx": idx,
        #         "score": score,
        #     },
        # )
        # batch_encoding["additional_data"] = {
        #     "idx": idx,
        #     "score": score,
        # }
        batch_encoding["idx"] = idx
        batch_encoding["score"] = score
        return batch_encoding


class PostProcessorStsb(PostProcessor):
    def __call__(self, decoded_prediction: list[str], batch: BatchEncoding) -> EvalInput:
        if not hasattr(batch, "score"):
            raise ValueError("In this case, BatchEncodingPlus must have score attribute")
        return EvalInput(
            [parse_s2f(pred) for pred in decoded_prediction],
            batch.score.tolist(),
        )


class EvaluatorStsb(Evaluator):
    def __init__(self) -> None:
        self.prediction_container: list[float] = []
        self.answer_container: list[float] = []

    def update(self, eval_input: EvalInput, **kwargs) -> None:
        for pred, label in zip(eval_input.predictions, eval_input.labels):
            self.prediction_container.append(pred)
            self.answer_container.append(label)

    def compute(self) -> dict[str, Any]:
        return {
            "pearson": pearsonr(self.prediction_container, self.answer_container)[0] * 100,
            "spearman": spearmanr(self.prediction_container, self.answer_container)[0] * 100,
        }

    def reset(self) -> None:
        self.prediction_container = []
        self.answer_container = []


def dataset_provider(
    task_type: Literal["STSB"], tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> tuple[DatasetFetcher, PreProcessor, PostProcessor, Evaluator]:
    if task_type == "STSB":
        return (
            DatasetFetcherStsb(),
            PreProcessorStsb(tokenizer, max_length=max_length),
            PostProcessorStsb(),
            EvaluatorStsb(),
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# class get_batch_encoding:
#     def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __call__(self, data_baggage: Seq2SeqDataBaggage) -> BatchEncoding:
#         batch_encoding: BatchEncoding = self.tokenizer(
#             data_baggage.prompt,
#             max_length=self.max_length,
#             truncation=True,
#         )
#         batch_encoding["labels"] = self.tokenizer.encode(
#             data_baggage.answer, max_length=self.max_length, truncation=True
#         )

#         # Register other information
#         batch_encoding["others"] = data_baggage.others
