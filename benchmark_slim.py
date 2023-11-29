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
#         lib_path = f"slim-dist/lib/{model}-{quantization}-cuda.so"
#         weight_path = f"slim-dist/weight/{model}-{quantization}"
#         compile_cmd = f"mlc_chat compile --model {model_path} --quantization {quantization} --output {lib_path}"
#         subprocess.run(shlex.split(compile_cmd))
#         # convert_weight_cmd = f"mlc_chat convert_weight --model {model_path} --quantization {quantization} --output {weight_path}"
#         # subprocess.run(shlex.split(convert_weight_cmd))
#         # gen_config_cmd = f"mlc_chat gen_mlc_chat_config --model {model_path} --quantization {quantization} --output {weight_path} --conv-template {conv_template}"
#         # subprocess.run(shlex.split(gen_config_cmd))
# print("build complete")

        
f = open("tmp.txt", "w")
for model, conv_template in model_conv_template.items():
    for quantization in quantization_list:
        model_path = "/models/" + model
        lib_path = f"slim-dist/lib/{model}-{quantization}-cuda.so"
        weight_path = f"slim-dist/weight/{model}-{quantization}"
        # Run benchmark
        f.write("Model: " + model + "\n")
        f.write("Quantization: " + quantization + "\n")
        f.flush()
        benchmark_cmd = f"""python -m mlc_chat.cli.benchmark --model {weight_path} --model-lib {lib_path} --generate-length 200 --device cuda --prompt "What's the meaning of life?" """
        subprocess.run(shlex.split(benchmark_cmd), stdout=f)