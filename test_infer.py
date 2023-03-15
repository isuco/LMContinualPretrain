import sys
sys.path.append('/data/FlagAI-master')
import torch
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "./checkpoints/40000"

model_dir = "/data/FlagAI/examples/gpt3_pretrain/state_dict"
auto_loader = AutoLoader(
    "lm",
    model_name="galactica-6.7b-en",
    model_dir=model_dir,
    )
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
#tok = AutoTokenizer.from_pretrained("/share/project/lijijie/new_tokens")
load_model_1 = torch.load("/data/FlagAI/examples/gpt3_pretrain/checkpoints/128/pytorch_model.bin")
model.load_state_dict(load_model_1, strict=True)
    
model.to(device)
model.eval()
predictor = Predictor(model, tokenizer)

test_data = [
    "where is beijing?",
    "how can we get to Boston?"
    "姚明多高?"
]

for text in test_data:
    print('===============================================\n')
    print(text)
    print(predictor.predict_generate_beamsearch(text))
    #print(tokenizer.encode_plus(text)['input_ids'][:-1])
    #print(tok.encode_plus(text))
