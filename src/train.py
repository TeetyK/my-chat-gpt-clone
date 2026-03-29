import torch
import torch.nn as nn
import torch.nn.functional as F
from src.GPTBERT import MyGPTBERT
from src.create_mask import create_masks
import torch

def train_model(model, dataloader, optimizer, criterion, device, pad_idx, vocab_size, num_epochs=3, save_path="my_llm_model.pth"):
    """
    ฟังก์ชันสำหรับเทรนโมเดล Encoder-Decoder (BERT+GPT)
    """
    print(f"🚀 เริ่มเทรนบน: {device} | จำนวนรอบ: {num_epochs} Epochs")
    
    # ดันโมเดลไปที่ GPU (ถ้ามี) และเปิดโหมด Training
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # วนลูปดึงข้อมูลจาก DataLoader ทีละ Batch
        for batch_idx, batch in enumerate(dataloader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # แบ่ง tgt เป็น Input และ เฉลย
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            # สร้าง Mask ปิดตา (ใช้ฟังก์ชัน create_masks ที่เราเคยเขียนไว้)
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
            
            # ล้างค่า Gradient เก่า
            optimizer.zero_grad()
            
            # 1. Forward Pass: ให้โมเดลลองเดา
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 2. คำนวณ Loss: ปรับรูปร่างเป็น 2 มิติเพื่อเทียบกับเฉลย 1 มิติ
            loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))
            
            # 3. Backward Pass: หาจุดที่ผิดพลาด
            loss.backward()
            
            # 4. Optimize: อัปเดตสมองโมเดล
            optimizer.step()
            
            total_loss += loss.item()
            
            # ปริ้นท์บอกความคืบหน้าทุกๆ 100 บัช (เพื่อไม่ให้เงียบเกินไป)
            if (batch_idx + 1) % 100 == 0:
                print(f"  - Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        # สรุปผลของแต่ละ Epoch
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}/{num_epochs} เสร็จสิ้น | Average Loss: {avg_loss:.4f}\n")
        
    # เมื่อเทรนครบทุกรอบ ให้เซฟน้ำหนักโมเดลเก็บไว้
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"💾 เซฟโมเดลไว้ที่ '{save_path}' เรียบร้อย พร้อมนำไปใช้งาน!")