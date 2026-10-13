# Save W0 + t*(W_merged - W0) for one adapter as a full HF model (for the export pipeline's merged/ dir).
# Same arithmetic as scaled_answers.py. Usage: scaled_merge.py <adapter> <out_dir> <t>
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
adapter, out, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
mid = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cpu")
w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
model = PeftModel.from_pretrained(model, adapter).merge_and_unload(); model.eval()
sd = model.state_dict(); n = 0
with torch.no_grad():
    for k, v in sd.items():
        if not torch.equal(v, w0[k]):
            v.copy_((w0[k].float() + t * (v.float() - w0[k].float())).to(v.dtype)); n += 1
print(n, "tensors scaled by", t, flush=True)
model.save_pretrained(out)
AutoTokenizer.from_pretrained(mid).save_pretrained(out)
