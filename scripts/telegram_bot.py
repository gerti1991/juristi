#!/usr/bin/env python3
"""
Simple Telegram Bot for Albanian Legal RAG System
Uses existing FastAPI backend for legal consultation
"""

import os
import sys
import logging
import requests
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Telegram bot imports
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ python-telegram-bot not installed. Run: pip install python-telegram-bot")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTelegramBot:
    """Simple Telegram bot that uses FastAPI backend"""
    
    def __init__(self):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot library not available")
        
        # Configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.admin_chat_id = os.getenv('TELEGRAM_ADMIN_CHAT_ID')
        self.fastapi_url = os.getenv('FASTAPI_BASE_URL', 'http://localhost:8000')
        
        if not self.bot_token:
            raise ValueError("❌ TELEGRAM_BOT_TOKEN not found in .env")
        
        logger.info(f"🤖 Bot initialized, FastAPI: {self.fastapi_url}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome = """⚖️ **Mirë se erdhe në Juristi AI\\!**

🤖 Unë jam asistenti juaj virtual për ligjet shqiptare\\.

📋 **Komandat:**
/start \\- Fillo bisedën
/help \\- Shfaq ndihmën  
/precise \\- Modalitet preciz
/analyze \\- Modalitet i analizuar

💡 **Ose thjesht shkruani pyetjen tuaj\\:**
"Cilat janë kushtet për divorc?"

⚠️ **Kujdes:** Ky është një këshillim automatik\\."""

        await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN_V2)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """🆘 **Juristi AI \\- Ndihmë**

🛠 **Komandat:**
/start \\- Fillo bisedën
/help \\- Shfaq ndihmën
/precise \\[pyetja\\] \\- Modalitet preciz
/analyze \\[pyetja\\] \\- Modalitet i analizuar

📝 **Shembuj:**
`/precise Cilat janë kushtet për divorc?`
`/analyze Shpjego dënimet penale për vjedhje`

Ose thjesht shkruani pyetjen tuaj direkt\\!"""

        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)
    
    async def precise_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /precise command"""
        query = ' '.join(context.args)
        if not query:
            await update.message.reply_text("Shkruani pyetjen pas komandës: `/precise Pyetja juaj`", parse_mode=ParseMode.MARKDOWN)
            return
        await self.process_query(update, context, query, "precise")
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        query = ' '.join(context.args)
        if not query:
            await update.message.reply_text("Shkruani pyetjen pas komandës: `/analyze Pyetja juaj`", parse_mode=ParseMode.MARKDOWN)
            return
        await self.process_query(update, context, query, "analyzed")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        query = update.message.text.strip()
        
        # Determine mode based on keywords
        analyze_keywords = ['analizë', 'shpjego', 'detaj', 'analyze', 'explain']
        mode = "analyzed" if any(word in query.lower() for word in analyze_keywords) else "precise"
        
        await self.process_query(update, context, query, mode)
    
    async def process_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, mode: str):
        """Process legal query using FastAPI"""
        chat_id = update.effective_chat.id
        
        # Send processing message
        processing_msg = await update.message.reply_text("🔍 Po përpunoj pyetjen tuaj...")
        
        try:
            # Call FastAPI
            endpoint = f"{self.fastapi_url}/analyse" if mode == "analyzed" else f"{self.fastapi_url}/search"
            response = requests.post(endpoint, json={"query": query}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('analysis' if mode == 'analyzed' else 'answer', 'Nuk u gjend përgjigje.')
                
                # Format response without special characters that break Markdown
                mode_text = "ANALIZUAR" if mode == "analyzed" else "PRECIZ"
                
                # Clean the answer text - remove problematic characters
                clean_answer = answer.replace('*', '').replace('_', '').replace('[', '').replace(']', '').replace('`', '')
                clean_query = query[:100].replace('*', '').replace('_', '').replace('[', '').replace(']', '').replace('`', '')
                
                response_text = f"""⚖️ JURISTI AI

📋 Pyetja: {clean_query}{'...' if len(query) > 100 else ''}

📖 Përgjigja ({mode_text}):
{clean_answer}

⚡ Gjeneruar: {datetime.now().strftime('%d/%m/%Y %H:%M')}

⚠️ Kujdes: Këshillim automatik. Konsultohuni me jurist për raste specifike."""
                
                # Delete processing message and send result (no parse_mode to avoid errors)
                await processing_msg.delete()
                await update.message.reply_text(response_text)
                
                logger.info(f"✅ Processed query for chat {chat_id}: {query[:50]}...")
            else:
                await processing_msg.edit_text(f"❌ Gabim teknik: Status {response.status_code}")
        
        except requests.exceptions.Timeout:
            try:
                await processing_msg.edit_text("⏳ Serveri po ngarkohet. Provoni përsëri pas 1-2 minutash.")
            except:
                await update.message.reply_text("⏳ Serveri po ngarkohet. Provoni përsëri pas 1-2 minutash.")
        except requests.exceptions.ConnectionError:
            try:
                await processing_msg.edit_text("❌ Nuk mund të lidhem me serverin. Sigurohuni që FastAPI është aktiv.")
            except:
                await update.message.reply_text("❌ Nuk mund të lidhem me serverin. Sigurohuni që FastAPI është aktiv.")
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            try:
                await processing_msg.edit_text(f"❌ Ndodhi një gabim: Ju lutem provoni përsëri.")
            except:
                await update.message.reply_text("❌ Ndodhi një gabim: Ju lutem provoni përsëri.")
    
    def run(self):
        """Run the bot"""
        logger.info("🚀 Starting Telegram bot...")
        
        # Test FastAPI connection
        try:
            response = requests.get(f"{self.fastapi_url}/", timeout=10)
            if response.status_code == 200:
                logger.info("✅ FastAPI backend is responding")
            else:
                logger.warning(f"⚠️ FastAPI status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to FastAPI: {e}")
            print("💡 Make sure to start FastAPI first: python main.py api")
            return
        
        # Create application
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("precise", self.precise_command))
        application.add_handler(CommandHandler("analyze", self.analyze_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Start bot
        logger.info("🟢 Bot is running... Press Ctrl+C to stop")
        application.run_polling(allowed_updates=["message", "callback_query"])

def main():
    """Main function"""
    try:
        bot = SimpleTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot failed: {e}")
        print(f"❌ Error: {e}")
        print("\n💡 Setup checklist:")
        print("1. TELEGRAM_BOT_TOKEN in .env file")
        print("2. FastAPI running: python main.py api")
        print("3. Dependencies: pip install python-telegram-bot")

if __name__ == "__main__":
    main()
