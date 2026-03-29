import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.multi_head_attention import MultiHeadAttention
from src.decoderblock import DecoderBlock
from src.encoderblock import EncoderBlock
class MyGPTBERT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_seq_len=512):
        super().__init__()
        
        # แปลงคำเป็นเวกเตอร์
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # ตำแหน่งของคำ (ใช้ Embedding แบบง่ายๆ แทนสมการ Sine/Cosine เพื่อความรวดเร็ว)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # สร้าง Encoder และ Decoder หลายๆ ชั้น (เช่น 6 ชั้น)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        
        # Layer สุดท้ายสำหรับทายคำศัพท์ (มีขนาดเท่ากับจำนวนคำศัพท์ในพจนานุกรมของเรา)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: ประโยคต้นทาง (ป้อนให้ BERT), tgt: ประโยคปลายทาง (ป้อนให้ GPT)
        seq_len_src = src.shape[1]
        seq_len_tgt = tgt.shape[1]
        device = src.device
        
        # สร้างเวกเตอร์บอกตำแหน่ง (0, 1, 2, 3...)
        pos_src = torch.arange(0, seq_len_src).unsqueeze(0).to(device)
        pos_tgt = torch.arange(0, seq_len_tgt).unsqueeze(0).to(device)
        
        # รวม Word Embedding กับ Positional Embedding
        enc_x = self.dropout(self.embedding(src) + self.pos_embedding(pos_src))
        dec_x = self.dropout(self.embedding(tgt) + self.pos_embedding(pos_tgt))
        
        # ไหลผ่าน BERT (Encoder)
        for layer in self.encoder_layers:
            enc_x = layer(enc_x, src_mask)
            
        # ไหลผ่าน GPT (Decoder) โดยพกข้อมูลจาก BERT (enc_x) ไปด้วย
        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x, src_mask, tgt_mask)
            
        # ทายคำศัพท์คำถัดไป!
        output = self.fc_out(dec_x)
        return output