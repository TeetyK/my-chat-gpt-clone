from datasets import load_dataset
from transformers import AutoTokenizer
from src.train import train_model
from src.translate import TranslationDataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.GPTBERT import MyGPTBERT
def translate_sentence(model, sentence, tokenizer,device, max_len=64):
    model.eval() 
    
    # 1. เตรียมประโยคอังกฤษให้ BERT (Encoder)
    src = tokenizer(sentence, return_tensors="pt", max_length=max_len, truncation=True)['input_ids'].to(device)
    src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
    
    with torch.no_grad():
        # ให้ฝั่ง BERT อ่านประโยคและสกัดความหมายออกมาเก็บไว้ใน enc_output
        enc_x = model.embedding(src) + model.pos_embedding(torch.arange(0, src.shape[1]).unsqueeze(0).to(device))
        for layer in model.encoder_layers:
            enc_x = layer(enc_x, src_mask)
        enc_output = enc_x
        
    # 2. เริ่มต้นฝั่ง GPT ด้วยคำว่า <s> (จุดเริ่มต้นประโยค)
    tgt_indexes = [tokenizer.bos_token_id]
    
    # 3. ลูปให้ GPT ทายคำถัดไปทีละคำ
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_indexes).unsqueeze(0).to(device)
        
        # สร้าง Mask ปิดตาอนาคตสำหรับ GPT
        seq_len = tgt_tensor.size(1)
        tgt_mask = torch.tril(torch.ones(1, seq_len, seq_len)).type(torch.bool).to(device)
        
        with torch.no_grad():
            # ไหลผ่าน Decoder (GPT)
            dec_x = model.embedding(tgt_tensor) + model.pos_embedding(torch.arange(0, seq_len).unsqueeze(0).to(device))
            for layer in model.decoder_layers:
                dec_x = layer(dec_x, enc_output, src_mask, tgt_mask)
            
            # ดูแค่คำตอบของ "คำสุดท้าย" ที่มันเพิ่งพ่นออกมา
            output = model.fc_out(dec_x)
            next_token = output.argmax(dim=-1)[:, -1].item() # เลือกคำที่มีความน่าจะเป็นสูงสุด
            
        # เอาคำที่ทายได้ไปต่อท้ายประโยค
        tgt_indexes.append(next_token)
        
        # ถ้ามันพ่นคำว่า </s> (จบประโยค) ออกมา ให้หยุดลูปทันที!
        if next_token == tokenizer.eos_token_id:
            break
            
    # แปลงตัวเลขกลับเป็นตัวหนังสือ
    translated_text = tokenizer.decode(tgt_indexes, skip_special_tokens=True)
    return translated_text

