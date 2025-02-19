import torch
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM

### GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:0")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
  

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.to(device)
model.eval()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
input_sentence = "Translate the following text from English into Italian:\nEnglish:The doctor asked the nurse to help her in the procedure.\Italian:"
inputs = tokenizer(input_sentence, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
  outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
org_hidden_states = torch.stack(outputs['hidden_states'])
input_shape = inputs['input_ids'].size()
_, seq_length = input_shape

# loop over tokens in the context, zeroing value vectors for each token by turn, and extracting alternative hidden_states for all tokens
scores = {}
_, seq_len = inputs['input_ids'].size()
heads = model.config.num_attention_heads
vz_matrix = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads, seq_len, seq_len)
for l, layer_module in enumerate(model.encoder.layer):
  for h in range(heads):
    for t in range(seq_len): # can be implemented without for but at the cost of memory when having long sequences, so I keep the loop for now
        extended_attention_mask: torch.Tensor = model.get_extended_attention_mask(inputs['attention_mask'], input_shape)
        alternative_layer_outputs = layer_module(
                            hidden_states=org_hidden_states[l], # previous layer's original output
                            attention_mask=extended_attention_mask,
                            value_zeroing_index=(h,t))
        # computing cosine distance  between each alternative token representation and its original to see how much others are affected in the absence of token t's value vector
        vz_matrix[l, h, :, t] = 1.0 - torch.nn.functional.cosine_similarity(org_hidden_states[l+1], alternative_layer_outputs[0], dim=-1)[0]
# normalizing to sum 1 for each row
scores['Value Zeroing'] = (vz_matrix / vz_matrix.sum(axis=-1, keepdims=True)).detach().cpu().numpy()
scores['Attention'] = torch.stack(outputs['attentions']).permute(1, 0, 2, 3, 4).squeeze(0).detach().cpu().numpy()

