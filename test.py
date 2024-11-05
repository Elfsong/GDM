from vllm import LLM, SamplingParams

pipe = LLM(model="01-ai/Yi-1.5-6B-Chat", tensor_parallel_size=2, gpu_memory_utilization=0.8, download_dir='/mnt/data/')

completion = pipe.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
)

print(completion[0].outputs[0].text)