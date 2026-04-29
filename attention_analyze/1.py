
import transformers

import os
os.chdir('..')

import modeling_albert



model = modeling_albert.AlbertRec.from_pretrained('checkpoints/albert_rec/checkpoint-3000')

# num_hidden_layers = 1
# num_hidden_groups = 1  
# inner_group_num = 1  
# num_attention_heads = 8

# model.albert_classifier.albert.encoder.albert_layer_groups
# ModuleList(
#   (0): AlbertLayerGroup(
#     (albert_layers): ModuleList(
#       (0): AlbertLayer(
#         (full_layer_layer_norm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
#         (attention): AlbertAttention(
#           (attention_dropout): Dropout(p=0, inplace=False)
#           (output_dropout): Dropout(p=0, inplace=False)
#           (query): Linear(in_features=512, out_features=512, bias=True)
#           (key): Linear(in_features=512, out_features=512, bias=True)
#           (value): Linear(in_features=512, out_features=512, bias=True)
#           (dense): Linear(in_features=512, out_features=512, bias=True)
#           (LayerNorm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
#         )
#         (ffn): Linear(in_features=512, out_features=2048, bias=True)
#         (ffn_output): Linear(in_features=2048, out_features=512, bias=True)
#         (activation): NewGELUActivation()
#         (dropout): Dropout(p=0, inplace=False)
#       )
#     )
#   )
# )

attention_module: transformers.modeling_albert.AlbertAttention = model.albert_classifier.albert.encoder.albert_layer_groups[0].albert_layers[0].attention

attention_weights = []

def create_hook(s):
    def hook(_module, _input, output):
        print('hook invoked')
        s.append(output[1].clone())
    return hook

handle = attention_module.register_forward_hook(create_hook(attention_weights))


