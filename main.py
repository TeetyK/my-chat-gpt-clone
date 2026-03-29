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
def main():

    # dataset = load_dataset("airesearch/scb_mt_enth_2020", split="train[:5000]", trust_remote_code=True)
    
    # dataset = load_dataset("opus100", "en-th", split="train[:5000]")
    # eng_data = [item['en'] for item in dataset['translation']]
    # thai_data = [item['th'] for item in dataset['translation']]

    # print(f"โหลดเสร็จแล้ว! ได้ประโยคมา {len(eng_data)} คู่")
    # print(f"🇬🇧 ตัวอย่าง EN: {eng_data[0]}")
    # print(f"🇹🇭 ตัวอย่าง TH: {thai_data[0]}")


    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # sample_tokens = tokenizer.encode(thai_data[0])
    # print(f"🔢 แปลงประโยคไทยเป็นตัวเลข: {sample_tokens}")
    # my_dataset = TranslationDataset(eng_data, thai_data, tokenizer, max_len=128)
    # dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # vocab_size = tokenizer.vocab_size
    # pad_idx = tokenizer.pad_token_id

    # model = MyGPTBERT(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4)
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # train_model(
    #     model=model, 
    #     dataloader=dataloader,
    #     optimizer=optimizer, 
    #     criterion=criterion, 
    #     device=device, 
    #     pad_idx=pad_idx, 
    #     vocab_size=vocab_size, 
    #     num_epochs=5, 
    #     save_path=".\\models\\my_translator_model.pth"
    # )

    # # ==========================================
    # # 🎯 ทดสอบการแปลภาษา!
    # # ==========================================
    # test_text = "Hello, how are you today?"
    # result = translate_sentence(model, test_text, tokenizer,device)

    # print(f"🇬🇧 Input: {test_text}")
    # print(f"🇹🇭 Output: {result}")

if __name__ == "__main__":
    main()
