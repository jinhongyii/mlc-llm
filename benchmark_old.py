import shlex
import subprocess

quantization_list = [
    # "q4f16_1",
    "q3f16_1",
    # "q4f32_1"
    ]
model_conv_template = {
    "Llama-2-7b-chat-hf": "llama-2",
    # "Mistral-7B-Instruct-v0.1": "mistral_default"
}

# for model, conv_template in model_conv_template.items():
#     for quantization in quantization_list:
#         model_path = "/models/" + model
#         compile_cmd = f"python3 build.py --debug-dump --model {model_path} --quantization={quantization} --use-cache=0 --no-cutlass-attn --no-cutlass-norm"
#         subprocess.run(shlex.split(compile_cmd))
# print("build complete")

        
f = open("tmp.txt", "w")
for model, conv_template in model_conv_template.items():
    for quantization in quantization_list:
        model_path = "/models/" + model
        lib_path = f"dist/{model}-{quantization}/{model}-{quantization}-cuda.so"
        weight_path = f"dist/{model}-{quantization}/params"
        # Run benchmark
        f.write("Model: " + model + "\n")
        f.write("Quantization: " + quantization + "\n")
        f.flush()
        benchmark_cmd = f"""python -m mlc_chat.cli.benchmark --model {weight_path} --model-lib {lib_path} --generate-length 200 --device cuda --prompt "What's the meaning of life?" """
        subprocess.run(shlex.split(benchmark_cmd), stdout=f)