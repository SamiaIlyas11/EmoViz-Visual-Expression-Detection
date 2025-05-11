import torch
from torch import nn
from .transformer import TemporalTransformer
import clip
import numpy as np

# Emotion class descriptions
EMOTION_DESCRIPTIONS = {
    "Angry": "A facial expression showing irritation and unrest, with a wrinkled forehead, narrowed eyes, and tight lips or a frown",
    "Disgust": "An expression of repulsion and displeasure, with a raised upper lip, a scrunched nose, and a downturned mouth",
    "Fear": "An expression of tension and withdrawal, with wide-open eyes, raised eyebrows, and a slightly open mouth. The face may appear physically tense or frozen in fear",
    "Happy": "An expression of contentment and pleasure, with a smile and the corners of the mouth turned up, often accompanied by crinkling around the eyes. The face may appear relaxed and at ease",
    "Neutral": "An expression of calm and neutrality, with a neutral mouth and no particular indication of emotion. The eyebrows are usually not raised or furrowed",
    "Sad": "An expression of sadness and sorrow, with a downturned mouth or frown, and sometimes tears or a tightness around the eyes. The face may appear physically withdrawn or resigned",
    "Surprise": "An expression of shock and astonishment, with wide-open eyes and raised eyebrows, sometimes accompanied by a gasp or an open mouth"
}

class VClip(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=4, dim_forward=2048):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Load CLIP model
        model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu", jit=False)
        for param in model.parameters():
            param.requires_grad = False
        self.backbone = model

        # Temporal Transformer for video encoding
        self.temporal = TemporalTransformer(
            input_dim=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_dim=dim_forward,
            dim_head=d_model // nhead
        )

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

        self.logit_scale = nn.Parameter(self.backbone.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True

        # Cache for emotion text features
        self.register_buffer('emotion_text_features', None)

    def _get_emotion_tokens(self):
        """
        Get tokenized emotion descriptions.
        Returns:
            torch.Tensor: Tokenized emotion descriptions
        """
        descriptions = list(EMOTION_DESCRIPTIONS.values())
        return clip.tokenize(descriptions, context_length=77, truncate=True)

    def _compute_emotion_text_features(self):
        """
        Compute and cache text features for emotion descriptions.
        """
        if self.emotion_text_features is None:
            with torch.no_grad():
                text_tokens = self._get_emotion_tokens().to(next(self.parameters()).device)
                text_features = self.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.emotion_text_features = text_features

    def encode_video(self, x):
        """
        Encodes a batch of video clips into feature representations.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input video data must be a torch.Tensor.")
        if len(x.shape) != 5 or x.shape[2] != 3:
            raise ValueError(f"Expected input shape (B, T, C, H, W), but got {x.shape}.")

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # Flatten batch and temporal dimensions
        features = self.backbone.encode_image(x)  # Shape: (B*T, d_model)
        features = features.reshape(B, T, -1)  # Reshape back to (B, T, d_model)
        features = self.temporal(features)  # Shape: (B, T, d_model)
        features = features[:, 0]  # Take the CLS token (first token)
        return features
        
    def encode_text(self, text):
        """
        Encodes text using CLIP's text encoder.
        """
        text_features = self.backbone.encode_text(text)
        return text_features

    def forward(self, x, text=None, mode='classification'):
        """
        Forward pass for video classification or video-text similarity tasks.
        Args:
            x (torch.Tensor): Video input of shape (B, T, C, H, W)
            text (Optional[torch.Tensor]): Tokenized text input for similarity mode
            mode (str): Either 'classification' or 'similarity'
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
            - In classification mode: returns logits (B, num_classes)
            - In similarity mode: returns (logits_per_image, logits_per_text)
        """
        video_features = self.encode_video(x)  # Shape: (B, d_model)
        
        if mode == 'classification':
            # Standard classification mode
            return self.classifier(video_features)
            
        elif mode == 'similarity':
            # CLIP-style similarity mode
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            
            # Use provided text or emotion descriptions
            if text is not None:
                text_features = self.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                self._compute_emotion_text_features()
                text_features = self.emotion_text_features
                
            # Compute similarity with temperature scaling
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * video_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            return logits_per_image, logits_per_text
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'classification' or 'similarity'.")
