# # ติดตั้งไลบรารี: pip install trl transformers

# import torch
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# from transformers import AutoTokenizer

# # 2. ตั้งค่า PPO (ตั้งค่า Learning Rate, Batch Size)
# config = PPOConfig(
#     model_name="my_llm_model",
#     learning_rate=1.41e-5,
#     batch_size=16
# )

# # 3. โหลดโมเดล LLM ของเรา (หุ้มด้วย Value Head เพื่อให้มันรับค่า Reward ได้)
# model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# # 4. โหลดหุ่นยนต์ตรวจข้อสอบ (Reward Model ที่เทรนมาแล้ว)
# # สมมติว่าเรามีฟังก์ชัน get_reward_from_model() เตรียมไว้แล้ว
# # reward_model = ...

# # 5. เรียกใช้ PPO Trainer (ตัวจัดการสมการคณิตศาสตร์สุดปวดหัวแทนเรา)
# ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

# print("🎮 เริ่มฝึก PPO...")

# # 6. Training Loop สไตล์ TRL
# for epoch, batch in enumerate(dataloader): # ดึงคำถามมา
#     queries = batch["input_ids"]
    
#     # ให้ LLM ของเราตอบคำถาม (Generation)
#     responses = ppo_trainer.generate(queries)
    
#     # ส่งคำตอบไปให้ Reward Model ตรวจให้คะแนน
#     rewards = [get_reward_from_model(q, r) for q, r in zip(queries, responses)]
    
#     # 💥 หัวใจหลัก: สั่งให้อัปเดตสมองด้วย PPO (จบในบรรทัดเดียว!)
#     stats = ppo_trainer.step(queries, responses, rewards)
    
#     print(f"Epoch {epoch} | Average Reward: {torch.mean(torch.tensor(rewards)):.2f}")

# model.save_pretrained("my_chatgpt_rlhf")