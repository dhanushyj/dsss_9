import logging
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer

TELEGRAM_BOT_TOKEN = "7861068411:AAFw6aRsz-sIidH7bZVNapzb8upt6aGa_Rk"

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing. Please set it correctly.")

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading TinyLlama model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
).to(device)

print(f"Model loaded successfully on {device}.")

async def start(update: Update, context):
    await update.message.reply_text("Hello! I am your AI Assistant powered by TinyLlama. Ask me anything!")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=200, temperature=0.7, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

async def handle_message(update: Update, context):
    user_text = update.message.text
    logging.info(f"Received message: {user_text}")
    ai_response = generate_response(user_text)
    logging.info(f"Generated response: {ai_response}")
    await update.message.reply_text(ai_response)

if __name__ == "__main__":
    try:
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("Bot is running...")
        app.run_polling()
    except Exception as e:
        print(f"Error starting the bot: {e}")
