import torch
import torch.nn as nn
import clip  # pip install git+https://github.com/openai/CLIP.git


class CLIPTextEncoder(nn.Module):
    """
    封装 CLIP 文本编码器，将文本描述编码为语义特征向量。
    
    输入: 文本列表 ["damaged building", "intact vegetation", ...]
    输出: 文本嵌入 (num_texts, embed_dim)
    """
    def __init__(self, clip_model="ViT-B/16", embed_dim=512, freeze=True,
                 pretrained_path=None):
        super().__init__()
        if pretrained_path is not None:
            # 从本地加载预训练权重
            self.clip_model, _ = clip.load(clip_model, device="cpu",
                                           download_root=pretrained_path)
        else:
            self.clip_model, _ = clip.load(clip_model, device="cpu")
        
        self.text_projection = nn.Linear(512, embed_dim)
        
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
    
    def forward(self, text_list):
        """
        Args:
            text_list: List[str], e.g. ["damaged building", "intact road", ...]
        Returns:
            text_features: (num_texts, embed_dim)
        """
        tokens = clip.tokenize(text_list).to(next(self.parameters()).device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        text_features = self.text_projection(text_features)  # (N, embed_dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
