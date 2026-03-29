import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # ใส่ Attention ที่เราสร้างไว้
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer Normalization 2 จุด
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network (Linear -> ReLU -> Linear)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # --- ส่วนที่ 1: Self-Attention ---
        # Encoder ใช้ข้อมูลตัวเองเป็นทั้ง Q, K, V (มองเพื่อนรอบข้างในประโยคเดียวกัน)
        attn_output = self.attention(x, x, x, mask)
        
        # บวกกลับด้วยข้อมูลเดิม (Residual) -> สุ่มดรอป -> ทำ Normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- ส่วนที่ 2: Feed Forward ---
        ff_output = self.ff(x)
        
        # ทำแบบเดิมอีกรอบ
        x = self.norm2(x + self.dropout(ff_output))
        
        return x