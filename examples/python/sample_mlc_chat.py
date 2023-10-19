from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout
from mlc_chat.chat_module import ChatConfig

# From the mlc-llm directory, run
# $ python examples/python/sample_mlc_chat.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1-tp1")
# You can change to other models that you downloaded, for example,
# cm = ChatModule(model="Llama-2-13b-chat-hf-q4f16_1")  # Llama2 13b model

# Generate a response for a given prompt
output = cm.benchmark_generate(
    prompt="What is the meaning of life?",
    generate_length=256
)

# Print prefill and decode performance statistics
print(f"Statistics: {cm.stats()}\n")

# output = cm.generate(
#     prompt="How many points did you list out?",
#     progress_callback=StreamToStdout(callback_interval=2),
# )

# Reset the chat module by
# cm.reset_chat()
