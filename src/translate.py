from torch.utils.data import Dataset, DataLoader
import torch
class TranslationDataset(Dataset):
    def __init__(self, english_sentences, thai_sentences, tokenizer, max_len):
        self.english = english_sentences
        self.thai = thai_sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.english)

    def __getitem__(self, idx):
        # 1. ดึงประโยคมาทีละคู่
        eng_text = str(self.english[idx])
        thai_text = str(self.thai[idx])

        # 2. แปลงข้อความเป็นตัวเลข (Tokenize) และเติมช่องว่างให้ยาวเท่ากัน (Padding)
        # ฝั่ง Source (ให้ BERT อ่าน)
        src_tokens = self.tokenizer.encode(eng_text, max_length=self.max_len, pad_to_max_length=True)
        
        # ฝั่ง Target (ให้ GPT อ่าน) เราต้องจัดการใส่ <SOS> และ <EOS>
        tgt_tokens = [self.tokenizer.sos_token_id] + self.tokenizer.encode(thai_text) + [self.tokenizer.eos_token_id]
        
        # ตัดหรือเติมให้ได้ความยาวเท่า max_len
        tgt_tokens = tgt_tokens[:self.max_len] 
        # ... (สมมติว่ามีโค้ดเติม Padding 0 ให้ครบโควต้า max_len) ...

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long)
        }

# สร้าง DataLoader เพื่อป้อนข้อมูลทีละ 32 ประโยค (Batch Size)
# my_dataset = TranslationDataset(eng_data, thai_data, tokenizer, max_len=128)
# dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)