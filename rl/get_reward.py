
def get_reward(generated_tokens, tokenizer):
    # แปลงตัวเลขกลับเป็นข้อความเพื่อตรวจ
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # กติกา: ถ้าประโยคลงท้ายด้วย "ครับ" หรือ "ค่ะ" เอาไปเลย 10 คะแนน!
    if text.strip().endswith("ครับ") or text.strip().endswith("ค่ะ"):
        return 10.0
    # ถ้าไม่มี เอาไป -1 คะแนน (เป็นการทำโทษเบาๆ)
    else:
        return -1.0