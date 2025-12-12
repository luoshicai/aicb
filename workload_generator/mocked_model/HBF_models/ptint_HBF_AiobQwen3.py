from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("./HBFAiobQwen3_config.json")
model = AutoModelForCausalLM.from_config(config)

for i, layer in enumerate(model.model.layers):
    print(f"\n===== Layer {i} =====")
    for name, module in layer.named_children():
        print("  ", name, " -> ", module.__class__.__name__)
