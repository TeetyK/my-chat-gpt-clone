import torch
import torch.nn as nn
import torch.nn.functional as F
from src.GPTBERT import MyGPTBERT
from src.create_mask import create_masks
def train():
    # ==========================================
    # 1. กำหนดค่าเริ่มต้นและสร้างโมเดล
    # ==========================================
    vocab_size = 5000  # สมมติว่าพจนานุกรมของเรามี 5,000 คำ
    pad_idx = 0        # ให้เลข 0 แทนช่องว่าง <PAD>
    batch_size = 2     # เทรนทีละ 2 ประโยคพร้อมกัน
    seq_len = 10       # ความยาวประโยคสูงสุด 10 คำ

    # สร้างโมเดลที่เราเขียนไว้ (โยนเข้า GPU ถ้ามี)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyGPTBERT(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4).to(device)

    # ==========================================
    # 2. จำลองข้อมูล (เหมือนเราดึงมาจาก DataLoader)
    # ==========================================
    # สุ่มตัวเลขตั้งแต่ 1 ถึง vocab_size (หลีกเลี่ยง 0 เพราะเป็น PAD)
    src_data = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    tgt_data = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)

    # ==========================================
    # 3. กำหนด Optimizer และ Loss Function
    # ==========================================
    # Adam คือตัวปรับน้ำหนักที่นิยมที่สุดสำหรับ Transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # CrossEntropyLoss คือตัววัดความผิดพลาด (บอกให้มันเมินคำที่เป็น <PAD> ไปซะ)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # ==========================================
    # 4. เริ่ม Training Loop (สมมติว่าฝึก 5 รอบ)
    # ==========================================
    print("🚀 เริ่มเทรนโมเดล...")
    model.train() # เปิดโหมด Training (เพื่อให้ Dropout และ LayerNorm ทำงาน)

    for epoch in range(5):
        optimizer.zero_grad() # ล้างค่า Gradient เก่าทิ้งก่อน
        
        # แบ่งข้อมูลฝั่ง GPT เป็น Input และ เฉลย (เลื่อนไป 1 ตำแหน่ง)
        tgt_input = tgt_data[:, :-1]   # เอาคำแรก ถึง ก่อนคำสุดท้าย
        tgt_expected = tgt_data[:, 1:] # เอาคำที่สอง ถึง คำสุดท้าย
        
        # สร้าง Mask
        src_mask, tgt_mask = create_masks(src_data, tgt_input, pad_idx)
        
        # 💥 FORWARD PASS: สั่งให้โมเดลคิด
        predictions = model(src_data, tgt_input, src_mask, tgt_mask)
        
        # ปรับรูปร่างข้อมูลก่อนคำนวณ Loss ให้เป็น 2 มิติ (Batch*SeqLen, VocabSize)
        loss = criterion(
            predictions.reshape(-1, vocab_size), 
            tgt_expected.reshape(-1)
        )
        
        # 💥 BACKWARD PASS: ให้โมเดลเรียนรู้จากความผิดพลาด
        loss.backward()
        
        # 💥 OPTIMIZE: อัปเดตสมอง (ปรับน้ำหนัก)
        optimizer.step()
        
        print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")

    print("✅ เทรนเสร็จสมบูรณ์!")