import os, re, threading, warnings
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

BASE = str(ROOT_DIR / "models" / "Llama-3.2-1B-Instruct")
ADAPTER = str(ROOT_DIR / "models" / "recipes-adapter-1B")

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=r"To copy construct from a tensor")

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

# load tokenizer and ensure pad_token exists (fallback to EOS)
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=dtype,
    attn_implementation="eager", # tells Hugging Face how to run the attention mechanism inside the model. eager is default.
    low_cpu_mem_usage=True,
    local_files_only=True,
)
model = (
    PeftModel.from_pretrained(base, ADAPTER, local_files_only=True)
    .merge_and_unload()
    .to(device)
    .eval()
)

cfg = GenerationConfig(
    max_new_tokens=500,
    do_sample=False,
    temperature=None,
    top_p=None,
    repetition_penalty=1.02,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
    use_cache=True, # reuse paste kv tokes. in transformers, every token depends on all previous tokens.
)

def bullets(s: str) -> str:
    parts = [p.strip() for p in re.split(r"[;,]\s*|\n+", s) if p.strip()]
    return "\n".join(f"- {p}" for p in parts)

def enforce_format(generated: str, input_ings: str) -> str:
    out = generated.strip()
    if "Used ingredients:" not in out:
        used = "Used ingredients:\n" + bullets(input_ings)
        if "Directions:" in out:
            parts = out.split("Directions:", 1)
            out = parts[0].rstrip() + "\n" + used + "\nDirections:" + parts[1]
        else:
            out = out + "\n\n" + used

    m = re.search(r"Directions:\n((?:\d+\..*\n?)*)", out) # Ensure directions has 5 steps
    if m:
        steps_block = m.group(1)
        steps = re.findall(r"^\d+\.\s", steps_block, flags=re.MULTILINE)
        if len(steps) < 5:
            fillers = ["Serve immediately.", "Enjoy warm.", "Plate and serve.", "Garnish if desired."]
            to_add = 5 - len(steps)
            extra = "\n".join(f"{len(steps)+i+1}. {fillers[i % len(fillers)]}" for i in range(to_add))
            out = out.rstrip() + ("\n" if not out.endswith("\n") else "") + extra + "\n"
    return out

def build_prompt(ings: str) -> str:
    fewshot_user = ( # giving example to model
        "Ingredients:\n- 2 eggs\n- 100 g flour\n- 50 g sugar\n\n"
        "Return exactly 1 recipe. Any number of ingredients from the list is allowed. "
        "Use ONLY the listed items. You may use additional ingredient: oil, butter, salt, pepper, water"
    )
    fewshot_assistant = (
        "Recipe:\n"
        "Title: Sweet Egg Pancake\n"
        "Used ingredients:\n"
        "- 2 eggs\n- 50 g sugar\n- 100 g flour\n"
        "Directions:\n"
        "1. Whisk 2 eggs with 50 g sugar until pale and slightly foamy (1–2 min).\n"
        "2. Sift in 100 g flour; whisk to a smooth pourable batter.\n"
        "3. Heat a dry nonstick pan on medium; pour a thin layer of batter.\n"
        "4. Cook 1–2 min until edges lift; flip and cook 20–30 sec.\n"
        "5. Repeat with remaining batter."
    )
    sys_msg = (
        "You are a recipe bot. Output exactly ONE recipe and nothing else.\n"
        "- Sections (in this exact order):\n"
        "  1) Recipe:\n"
        "  2) Title: <title>\n"
        "  3) Used ingredients:\n"
        "     - <copy each from input with quantities>\n"
        "  4) Directions:\n"
        "     1. ...\n"
        "     2. ...\n"
        "     3. ...\n"
        "     4. ...\n"
        "     5. ...\n"
        "- Never omit the 'Used ingredients' section.\n"
        "- Directions must have exactly 5 numbered steps.\n"
        "- Use ONLY the user’s listed ingredients; you may also use oil, butter, salt, pepper, and water."
    )
    user_msg = (
        f"Ingredients:\n{bullets(ings)}\n\n"
        "Return exactly one recipe in the required format above. "
        "Include the 'Used ingredients' section copied from the input, and exactly 5 numbered steps."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": fewshot_user},
        {"role": "assistant", "content": fewshot_assistant},
        {"role": "user", "content": user_msg},
    ]
    from transformers import PreTrainedTokenizerBase
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # tok is the loaded tokenizer
    return prompt

def generate_recipe(ings: str) -> str:
    prompt = build_prompt(ings)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096) # make encodings (kv) using tokenizer, instruct supports up to 4096
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad(): # saves memory and enhances speed
        out = model.generate(**enc, generation_config=cfg)
    gen_ids = out[0][enc["input_ids"].shape[1]:] # only keep new generated part
    text = tok.decode(gen_ids, skip_special_tokens=True).strip() # string
    return enforce_format(text, ings)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Recipe Generator")
        self.geometry("760x560")
        self.minsize(680, 520)

        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(frm_top, text="Ingredients (comma-separated):").pack(anchor="w")
        self.entry = ttk.Entry(frm_top)
        self.entry.pack(fill="x", pady=6)
        self.entry.insert(0, "3 eggs, 200 g flour, 150 g sugar")

        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", padx=12, pady=(0, 6))
        self.btn = ttk.Button(frm_btn, text="Generate Recipe", command=self.on_generate)
        self.btn.pack(side="left")
        self.status = ttk.Label(frm_btn, text=f"Ready on {device.upper()}")
        self.status.pack(side="right")

        frm_out = ttk.Frame(self)
        frm_out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(frm_out, wrap="word")
        yscroll = ttk.Scrollbar(frm_out, command=self.txt.yview)
        self.txt.configure(yscrollcommand=yscroll.set)
        self.txt.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.entry.bind("<Return>", lambda e: self.on_generate())

    def on_generate(self):
        ings = self.entry.get().strip()
        if not ings:
            messagebox.showinfo("Recipe Generator", "Please enter some ingredients.")
            return
        self.btn.config(state="disabled")
        self.status.config(text="Generating…")
        self.txt.delete("1.0", "end")

        def worker(): # makes recipe generation seperate of interaction with the GUI. freezes otherwise
            try:
                text = generate_recipe(ings)
            except Exception as e:
                text = f"Error: {e}"
            self.after(0, lambda: self.show_result(text))

        threading.Thread(target=worker, daemon=True).start() # start new background thread running worker()

    def show_result(self, text: str):
        self.txt.insert("1.0", text + "\n")
        self.btn.config(state="normal")
        self.status.config(text=f"Ready on {device.upper()}")

if __name__ == "__main__":
    app = App()
    app.mainloop()