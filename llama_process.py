import os
from modal import Image, Secret, Stub, method, Mount
import csv
import json

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-13b-chat-hf",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

MODEL_DIR = "/model"

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("my-huggingface-secret"),
        timeout=60 * 20,
    )
)

stub = Stub("example-vllm-inference", image=image)

@stub.cls(gpu="A100", secret=Secret.from_name("my-huggingface-secret"), mounts=[Mount.from_local_file("train-v2.0.json", remote_path="/root/train-v2.0.json")])
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model.
        self.llm = LLM(MODEL_DIR)
        self.template = """SYSTEM: You are a helpful assistant.
USER: {}
ASSISTANT: """

        print(os.listdir("/root/"))
        with open("train-v2.0.json", "r") as file:
            data = json.load(file)
     
        output_data = []
        
        

    @method()
    def generate(self, question):
        from vllm import SamplingParams

        prompt = self.template.format(question)
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompt, sampling_params)
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")

@stub.local_entrypoint()
def main():
    model = Model()