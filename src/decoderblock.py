import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.multi_head_attention import MultiHeadAttention
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Self-Attention สำหรับตัว Decoder เอง
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # 2. Cross-Attention สำหรับเชื่อมกับ Encoder
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalization 3 จุด เพราะมี Attention 2 ตัว
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # --- ส่วนที่ 1: Masked Self-Attention ---
        # สังเกตว่าเราใช้ tgt_mask (Target Mask) เพื่อปิดตาไม่ให้มองเห็นคำในอนาคต
        attn1 = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        
        # --- ส่วนที่ 2: Cross-Attention (จุดเชื่อมต่อ BERT กับ GPT) ---
        # Q (Query) มาจากฝั่ง GPT (x)
        # K, V (Key, Value) มาจากฝั่ง BERT (enc_output)
        attn2 = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        
        # --- ส่วนที่ 3: Feed Forward ---
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x