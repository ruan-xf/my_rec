
import matplotlib.pyplot as plt
import torch

attention_weight = torch.load('attention_weight.pt')

# # 某个 head 的完整注意力矩阵

for attn in attention_weight:
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.show()



# avg = attention_weight.mean(dim=0).cpu().numpy()  # [74, 74]
# plt.imshow(avg, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.show()