from datasets import load_dataset
from transformers import AutoTokenizer
def main():

    # dataset = load_dataset("airesearch/scb_mt_enth_2020", split="train[:5000]", trust_remote_code=True)
    dataset = load_dataset("opus100", "en-th", split="train[:5000]")
    eng_data = [item['en'] for item in dataset['translation']]
    thai_data = [item['th'] for item in dataset['translation']]

    print(f"โหลดเสร็จแล้ว! ได้ประโยคมา {len(eng_data)} คู่")
    print(f"🇬🇧 ตัวอย่าง EN: {eng_data[0]}")
    print(f"🇹🇭 ตัวอย่าง TH: {thai_data[0]}")


    print("\n⏳ กำลังโหลด Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    sample_tokens = tokenizer.encode(thai_data[0])
    print(f"🔢 แปลงประโยคไทยเป็นตัวเลข: {sample_tokens}")


if __name__ == "__main__":
    main()
