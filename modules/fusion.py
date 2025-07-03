import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch_geometric.nn import GATConv, global_mean_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FusionLayer')

class CrossModalAttention(nn.Module):
    """Attention mechanism for modality fusion"""
    
    def __init__(self, feature_dims: dict, embed_dim: int = 512):
        super().__init__()
        self.visual_proj = nn.Linear(feature_dims['visual'], embed_dim)
        self.text_proj = nn.Linear(feature_dims['text'], embed_dim)
        self.audio_proj = nn.Linear(feature_dims['audio'], embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(embed_dim * 3, embed_dim)
    
    def forward(self, visual_feats, text_feats, audio_feats):
        # Project features to common space
        V = self.visual_proj(visual_feats)
        T = self.text_proj(text_feats)
        A = self.audio_proj(audio_feats)
        
        # Combine into sequence
        combined = torch.cat([V.unsqueeze(1), T.unsqueeze(1), A.unsqueeze(1)], dim=1)
        
        # Apply attention
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Aggregate and fuse
        fused = self.fc(attn_output.view(attn_output.size(0), -1))
        return fused

class GraphFusion(nn.Module):
    """Graph-based fusion using GNN"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

class HybridFusion:
    """Hybrid fusion layer combining attention and GNN"""
    
    def __init__(self):
        self.feature_dims = {
            'visual': 1792,
            'text': 768,
            'audio': 80
        }
        self.attention = CrossModalAttention(self.feature_dims)
        self.graph_fusion = GraphFusion(512)
    
    def build_graph(self, visual_feats, text_feats, audio_feats):
        """Construct graph from multimodal features"""
        # This would implement graph construction logic
        # Placeholder implementation
        num_nodes = len(visual_feats) + len(text_feats) + len(audio_feats)
        x = torch.randn(num_nodes, 512)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
        batch = torch.zeros(num_nodes, dtype=torch.long)
        return x, edge_index, batch
    
    def fuse(self, visual_results, text_results, audio_results):
        """Fuse multimodal features"""
        try:
            # Attention-based fusion
            visual_feats = torch.tensor(np.array(visual_results['frame_features']))
            text_feats = torch.tensor(np.array(text_results['text_features']))
            audio_feats = torch.tensor(np.array(audio_results['segment_embeddings']))
            
            attn_fused = self.attention(
                visual_feats.mean(dim=0),
                text_feats.mean(dim=0),
                audio_feats.mean(dim=0)
            )
            
            # Graph-based fusion
            x, edge_index, batch = self.build_graph(
                visual_results['frame_features'],
                text_results['text_features'],
                audio_results['segment_embeddings']
            )
            graph_out = self.graph_fusion(x, edge_index, batch)
            
            # Weighted fusion
            attention_weight = 0.6
            graph_weight = 0.4
            fused_confidence = (attention_weight * attn_fused.mean() + 
                              graph_weight * graph_out.item())
            
            return {
                'fused_confidence': float(fused_confidence),
                'attention_weights': attn_fused.detach().numpy(),
                'graph_output': graph_out.item()
            }
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            # Fallback to average confidence
            confidences = [
                visual_results.get('visual_confidence', 0.5),
                text_results.get('text_confidence', 0.5),
                audio_results.get('audio_confidence', 0.5)
            ]
            return {'fused_confidence': float(np.mean(confidences))}