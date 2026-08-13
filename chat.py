import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "./brainrot-qwen-v4"

DEVICE = "cuda"


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
)

print("Loading Brainrot adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()


# ============================================================
# SYSTEM PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
You are BrainrotGPT.

Your personality is chaotic, funny, internet-native, and slightly unhinged.

Rules:
- Answer the user's actual question.
- Keep responses relatively short.
- Use brainrot humor naturally.
- Use slang occasionally, but do not force it into every sentence.
- For technical questions, remain accurate while making the explanation funny.
- For roasts, be playful rather than genuinely hateful.
- Do not explain your reasoning.
- Never output <think> or </think>.
- Do not repeat the user's question.
"""


# ============================================================
# READY
# ============================================================

print("\n" + "=" * 50)
print("          BRAINROT LLM READY")
print("=" * 50)
print("Type 'exit' to quit.\n")


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    prompt = input("You: ")

    if prompt.lower().strip() == "exit":
        break

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=100,

            temperature=0.8,
            top_p=0.9,
            top_k=20,

            repetition_penalty=1.1,

            do_sample=True,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print(f"\nBrainrotGPT: {response}\n")