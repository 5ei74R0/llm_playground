"""
Entrypoint of `llm_playground`.
"""
import fire


def main(
    dataset: str,
    model: str,
):
    print(f"dataset: {dataset}")
    print(
        f"model: {model}"
    )


if __name__ == "__main__":
    fire.Fire(main)
