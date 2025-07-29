import warnings
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, \
                        TextStreamer, AutoModelForCausalLM, AutoTokenizer, \
                        AutoConfig
import gc
from copy import deepcopy
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_nparams(model):
    """Calculate teh total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")

fix_torch_seed()

# Model configuration
config = LlamaConfig()
print(config)

# update parameters to change the model architecture
config.num_hidden_layers = 12      # num of transformers blocks stack one after another
config.hidden_size = 1024          # the width of every token embedding
config.intermediate_size = 4096    # the inner dimension of the ffn inside each block
config.num_key_value_heads = 8     # the count of K/V heads that are actually stored in attention
config.torch_dtype = "bfloat16"    # tell transformers to initialise weights in brain-float-16
config.use_cache = False           # disable KV cache during model.generate()
print(config)

# ---------------------------------------------------------------------------
# 1. Ways of Weight Initialisation Exploration
# ---------------------------------------------------------------------------
# 1.a Random weight initialization
model = LlamaForCausalLM(config)
print("1.A. Random weight initialisation — initial model configs: ")
print(model)
print_nparams(model)

# Inspect a sample layer from the model
layer_name = "model.layers.0.self_attn.q_proj.weight"
for name, param in model.named_parameters():
    if name == layer_name:
        print(f"First 30 weights of layer '{layer_name}':")
        print(param.data.view(-1)[:30])
        break

# Load a pre-trained tokenizer from Upstage Solar, compatible with Llama-2 tokenizer
model_dir = "upstage/SOLAR-10.7B-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Run a simple inference with prompt on this prepared model
prompt = "I am an engineer. I love"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

streamer = TextStreamer(
    tokenizer=tokenizer,
    skip_prompt=True,        # don't print the prompt again.
    skip_special_tokens=True # hides <s> </s> etc.
)

outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False      # greedy_decoding
) # get nonsense output since the current model's weights is randomly generated

# remove the model from memory to avoid crashing the kernel
del model      # remove the last python references to model, streamer, output
del streamer
del outputs
gc.collect()  # force the CPython garbage-collector to run right now, ensure the memory is returned immediately

# 1.b. reuse general pretrained model weights
model_name_or_path = "upstage/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",        # device_map triggers automatic chunk-wise weight loading and off-loading logic that lives in the Accelerate library
    torch_dtype=torch.bfloat16,
)
print("1.B. Reusing general pretrained model weights: ")
print(model)
print_nparams(model)

del model
gc.collect()

# 1.c Downscaling from a general pretrained model
model_name_or_path = "upstage/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("1.C. Downscaling from a pretrained model — initial model configs before downgrade: ")
print(model)
print_nparams(model)

# remove the middle 2 layers (layer 5 and 6) and update the config
layers = model.model.layers
model.model.layers = layers[:5] + layers[-5:]

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_hidden_layers=len(model.model.layers),
)
model.config = config

print("1.C. Downscaling from a pretrained model — model configs after downgrade: ")
print(model)
print_nparams(model)

del model
gc.collect()

# 1.D. Upscaling an existing model
# here we upscale Upstage's tinySolar-248m-4k model from 12 layers to 16 layers
# We'll use this method for pretraining in this project
config = LlamaConfig(
    num_hidden_layers=16,    # we want the model to have 16 layers
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=32,  # num of q's head
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    use_cache=False,
)
model = LlamaForCausalLM(config) # skeleton of the model
print("1.D. Upscaling from a pretrained model — initial model configs before upgrade: ")
print(model)
model = model.to(dtype=torch.bfloat16)
print_nparams(model)

model_name_or_path = "upstage/TinySolar-248m-4k" # the 12 layer checkpoint
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print("1.D. Upscaling from a pretrained model — pretrained model configs: ")
print(pretrained_model)
print_nparams(pretrained_model)

model.model.layers = deepcopy(pretrained_model.model.layers[:-4])   \
    + deepcopy(pretrained_model.model.layers[4:])  # the top & bottom eight

# load these layers to match the tokenizer vocab, reuse them to make the new model speaks the same language straight away
model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)  # input word-embedding matrix
model.lm_head = deepcopy(pretrained_model.lm_head)  # the output projection

print("1.D. Upscaling from a pretrained model — model configs after upgrade: ")
print(model.config)
print_nparams(pretrained_model)

# Run a simple inference on the prepared model
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(
    tokenizer=tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False
)

# save the prepared model to disk
model.save_pretrained("./data/TinySolar-308m-4k-init")

