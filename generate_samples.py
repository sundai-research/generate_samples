#!/usr/bin/env python3
import typer
import asyncio
import json
import httpx
from datasets import load_dataset
from transformers import AutoTokenizer
# from bert_score import score
from bert_score import BERTScorer


scorer = BERTScorer(lang="en")
# Default generation settings
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 31319

app = typer.Typer()


def reward_fn(text: str) -> float:
    """
    Bert-based reward function to evaluate the quality of generated text.

    """
    hard_coded_answer="There is a chronic need for more housing for prison leavers in Wales, according to a charity."
    # print(text)
    # exit
    # P, R, F1 = score([text], [hard_coded_answer], lang="en")
    # print(text)
    P, R, F1 = scorer.score([text], [hard_coded_answer])

   

    return F1.item()

@app.command()
def generate(
    input_path: str = typer.Option(..., help="Path to input JSONL file"),
    output_path: str = typer.Option(..., help="Path to output JSONL file"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, "--model-name", "-m", help="Model name or path to load tokenizer"),
    num_proc: int = typer.Option(1, "--num-proc", "-p", help="Number of parallel processes"),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Generation endpoint host"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Generation endpoint port"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    max_new_tokens: int = typer.Option(8192, "--max-new-tokens", help="Maximum number of new tokens to generate"),
    enable_thinking: bool = typer.Option(True, "--thinking/--no-thinking", help="Enable thinking mode"),
    num_responses_per_question: int = typer.Option(32, "--num-responses-per-question", "-n", help="Number of responses per question"),
):
    """Generate answers for each example in a JSONL dataset using a local model endpoint."""
    # Load tokenizer for specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Asynchronous helper to generate one response
    async def _generate_one(text: str) -> str:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"http://{host}:{port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                    },
                    "return_logprob": True,
                },
            )
        res_json = response.json()
        text = res_json.get("text", "")
        reward = reward_fn(text)
        print(reward)
        return {"text": text, "reward": reward}
        # return res_json

    # Load JSONL dataset and prepare examples
    dataset = load_dataset("json", data_files=input_path, split="train")
    examples = list(dataset)

    # Build and run async tasks
    tasks = [
        _generate_one(example.get("text", ""))
        for example in examples
        for _ in range(num_responses_per_question)
    ]
    # Wrap gather in a coroutine so asyncio.run receives a coroutine
    async def _run_all_tasks():
        return await asyncio.gather(*tasks)
    responses = asyncio.run(_run_all_tasks())
    # Assemble output objects
    output_objs = []
    idx = 0
    for example in examples:
        for _ in range(num_responses_per_question):
            output_objs.append({
                "text": example.get("text", ""),
                "answer": responses[idx]["text"],
                "reward": responses[idx]["reward"],
            })
            idx += 1

    # Write to output JSONL
    with open(output_path, "w") as f:
        for obj in output_objs:
            f.write(json.dumps(obj) + "\n")
    typer.echo(f"Generation complete. Output saved to {output_path}")

if __name__ == "__main__":
    app()

