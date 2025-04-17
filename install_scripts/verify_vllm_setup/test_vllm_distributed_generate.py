from vllm import LLM, SamplingParams
import torch.distributed as dist
import os
import torch

def main():
    """
    Run vLLM in a distributed setting with 8 GPUs on a single node using torchrun
    """
    # Get distributed environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Print basic distributed info - each process will print this
    print(f"Initializing process {rank}/{world_size} (local_rank: {local_rank})")

    # Set the device for this process
    torch.cuda.set_device(local_rank)

    # Initialize the distributed process group - needed for NCCL communication
    dist.init_process_group(backend="nccl")

    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Initialize the LLM with distributed settings
    llm = LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=2,  # Use all available GPUs
        distributed_executor_backend="external_launcher",  # Tell vLLM we're using torchrun
        gpu_memory_utilization=0.8,
        dtype="half",
        trust_remote_code=True,
    )

    # All processes need to participate in generation since model is sharded
    outputs = llm.generate(prompts, sampling_params)

    # But only rank 0 should print to avoid duplicate output
    if rank == 0:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Make sure all processes synchronize before finishing
    dist.barrier()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
