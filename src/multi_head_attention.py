import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # สร้าง Weight Matrices สำหรับ Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Layer สุดท้ายสำหรับรวมผลลัพธ์จากทุกหัวเข้าด้วยกัน
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. โยนข้อมูลเข้า Linear Layer แล้วหั่นแบ่งเป็นหลายๆ หัว (Multi-Head)
        # .view() ใช้ปรับรูปร่าง .transpose() ใช้สลับตำแหน่งแกน
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. คำนวณค่า Attention (Q คูณ K Transpose แล้วหารด้วยรากที่สองของ d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # ถ้ามีการใส่ Mask (เช่นฝั่ง Decoder หรือปิด Padding) ให้แทนที่ด้วยค่าติดลบเยอะๆ
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # ใช้ Softmax เพื่อแปลงคะแนนให้เป็นความน่าจะเป็น (รวมกันได้ 1)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 3. เอาคะแนนที่ได้ไปคูณกับ V (Value)
        output = torch.matmul(attention_weights, V)
        
        # 4. รวมร่างทุกหัวกลับมาเป็นก้อนเดียว
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # โยนเข้า Linear Layer สุดท้าย
        return self.W_o(output)