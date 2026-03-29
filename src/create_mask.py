import torch
def create_masks(src, tgt, pad_idx=0):
    # 1. Source Mask สำหรับ BERT (ปิดตาเฉพาะช่องว่าง <PAD>)
    # รูปร่าง: (batch_size, 1, 1, seq_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 2. Target Mask สำหรับ GPT (ปิดช่องว่าง + ปิดอนาคต)
    # เริ่มจากปิดช่องว่างก่อน
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    
    # สร้างสามเหลี่ยมแห่งการปิดตา (Lower Triangular Matrix)
    # torch.tril จะทำให้ค่าด้านบนขวาของเมทริกซ์เป็น 0 (มองไม่เห็นอนาคต)
    seq_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones(1, seq_len, seq_len)).type_as(src_mask)
    
    # เอาสองเงื่อนไขมารวมกัน (ต้องไม่ใช่ <PAD> และ ต้องไม่อยู่ในอนาคต)
    tgt_mask = tgt_pad_mask & nopeak_mask
    
    return src_mask, tgt_mask