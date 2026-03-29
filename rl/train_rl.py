import torch.optim as optim
from torch.distributions import Categorical
import torch
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
from src.translate_sentence import translate_sentence
from rl.get_reward import get_reward
def train_rl(model ,tokenizer , device):
    rl_optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print("🎮 เริ่มฝึกโมเดลด้วย Reinforcement Learning!")
    model.train()

    for episode in range(100):
        rl_optimizer.zero_grad()
        
        prompt_text = "Hello"
        src = tokenizer(prompt_text, return_tensors="pt")['input_ids'].to(device)
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
        
        enc_x = model.embedding(src) + model.pos_embedding(torch.arange(0, src.shape[1]).unsqueeze(0).to(device))
        for layer in model.encoder_layers:
            enc_x = layer(enc_x, src_mask)
        enc_output = enc_x

        tgt_indexes = [tokenizer.bos_token_id]
        log_probs = [] # ตัวแปรสำคัญสำหรับทำ RL!
        
        for i in range(20): # สมมติให้แต่งยาวสุด 20 คำ
            tgt_tensor = torch.tensor(tgt_indexes).unsqueeze(0).to(device)
            seq_len = tgt_tensor.size(1)
            tgt_mask = torch.tril(torch.ones(1, seq_len, seq_len)).type(torch.bool).to(device)
            
            # ให้โมเดลพ่นความน่าจะเป็นของคำถัดไปออกมา
            dec_x = model.embedding(tgt_tensor) + model.pos_embedding(torch.arange(0, seq_len).unsqueeze(0).to(device))
            for layer in model.decoder_layers:
                dec_x = layer(dec_x, enc_output, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_x)[:, -1, :] # เอาเฉพาะคำล่าสุด
            
            # 💥 ความต่างของ RL: เราไม่เลือกคำที่โอกาสสูงสุด (argmax) 
            # แต่เรา "สุ่ม (Sample)" ตามเปอร์เซ็นต์ เพื่อให้มันได้ลองผิดลองถูก (Exploration)
            m = Categorical(logits=logits)
            action = m.sample()
            
            tgt_indexes.append(action.item())
            log_probs.append(m.log_prob(action)) # เก็บค่า Log Prob ไว้คำนวณ Loss
            
            if action.item() == tokenizer.eos_token_id:
                break

        # 5. สิ้นสุดการแต่งประโยค -> ให้คะแนน (Reward)
        reward = get_reward(tgt_indexes, tokenizer)
        
        # 6. คำนวณ RL Loss (สมการ Policy Gradient: Loss = - (LogProb * Reward))
        # ยิ่ง Reward เป็นบวกเยอะ ค่า Loss จะยิ่งติดลบเยอะ (โมเดลชอบ)
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * reward)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        # 7. ปรับสมอง (Backpropagate)
        policy_loss.backward()
        rl_optimizer.step()
        
        if (episode + 1) % 10 == 0:
            generated_text = tokenizer.decode(tgt_indexes, skip_special_tokens=True)
            print(f"Episode {episode+1} | Reward: {reward} | Output: {generated_text}")

    print("✅ ฝึกเสร็จแล้ว! ตอนนี้โมเดลน่าจะพยายามพูดคำว่า 'ครับ' หรือ 'ค่ะ' มากขึ้นแล้ว")