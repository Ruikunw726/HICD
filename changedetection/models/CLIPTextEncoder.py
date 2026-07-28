import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """
    使用 open_clip 加载 CLIP ViT-B/16 文本编码器。

    输入: 文本列表 ["damaged building", "intact vegetation", ...]
    输出: 文本嵌入 (num_texts, embed_dim)
    """
    def __init__(self, clip_model="ViT-B-16", embed_dim=512, freeze=True,
                 pretrained_path=None):
        super().__init__()

        import open_clip
        import logging
        logging.disable(logging.WARNING)  # suppress pretrained warning

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=""
            )
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            self.clip_model = model
        else:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai"
            )

        logging.disable(logging.NOTSET)  # restore logging

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.text_projection = nn.Linear(512, embed_dim)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, text_list):
        """
        Args:
            text_list: List[str]
        Returns:
            text_features: (num_texts, embed_dim)
        """
        tokens = self.tokenizer(text_list).to(next(self.parameters()).device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        text_features = self.text_projection(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
