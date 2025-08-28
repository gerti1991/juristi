# 🤖 Telegram Bot Setup Guide for Juristi AI

This guide will help you set up the Telegram bot integration for your Albanian Legal RAG system.

## 📋 Prerequisites

- Python environment with Juristi AI installed
- FastAPI backend running (`python main.py api`)
- Telegram account

## 🚀 Step-by-Step Setup

### Step 1: Create Your Telegram Bot

1. **Open Telegram** and search for `@BotFather`
2. **Start a chat** with BotFather and send `/start`
3. **Create a new bot** by sending `/newbot`
4. **Choose a name** for your bot (e.g., "Juristi AI Legal Assistant")
5. **Choose a username** for your bot (e.g., "juristi_ai_bot")
6. **Copy the bot token** - it looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### Step 2: Get Your Chat ID (Optional but Recommended)

1. **Start a chat** with your new bot by clicking the link BotFather provided
2. **Send any message** to your bot (e.g., "Hello")
3. **Open this URL** in your browser: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Replace `<YOUR_BOT_TOKEN>` with your actual token
4. **Find your chat ID** in the JSON response - look for `"chat":{"id":123456789}`

### Step 3: Configure Environment Variables

Edit your `.env` file and add these lines:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ADMIN_CHAT_ID=your_chat_id_here
TELEGRAM_BOT_USERNAME=your_bot_username
TELEGRAM_WEBHOOK_URL=https://your-domain.com/webhook/telegram

# Telegram Bot Settings
TELEGRAM_BOT_NAME=Juristi AI
TELEGRAM_WELCOME_ENABLED=true
TELEGRAM_HELP_ENABLED=true
TELEGRAM_ADMIN_NOTIFICATIONS=true
```

**Example:**
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_ADMIN_CHAT_ID=123456789
TELEGRAM_BOT_USERNAME=juristi_ai_bot
TELEGRAM_BOT_NAME=Juristi AI Legal Assistant
```

### Step 4: Install Dependencies

```bash
pip install python-telegram-bot==21.0
```

### Step 5: Start Your Bot

1. **Start FastAPI backend first:**
   ```bash
   python main.py api
   ```

2. **In another terminal, start the Telegram bot:**
   ```bash
   python main.py telegram
   ```

## ✅ Testing Your Bot

1. **Find your bot** on Telegram by searching for `@your_bot_username`
2. **Send `/start`** to begin the conversation
3. **Try some commands:**
   - `/help` - Show available commands
   - `/precise What are the conditions for divorce?` - Quick legal question
   - `/analyze Explain the criminal penalties for theft` - Detailed analysis
   - Or just send a question directly: "Cilat janë kushtet për divorc?"

## 🛠 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Start conversation with bot | `/start` |
| `/help` | Show help and available commands | `/help` |
| `/precise <question>` | Quick, precise legal answer | `/precise Kushtet për divorc?` |
| `/analyze <question>` | Detailed legal analysis | `/analyze Shpjego dënimet penale` |

## 🔧 Troubleshooting

### Bot Token Error
- Make sure you copied the token correctly from BotFather
- Ensure there are no extra spaces in the .env file

### FastAPI Connection Error
- Make sure FastAPI is running: `python main.py api`
- Check that the URL is correct: `http://localhost:8000`

### Bot Not Responding
- Check the terminal for error messages
- Verify your bot token is valid
- Ensure dependencies are installed: `pip install python-telegram-bot`

## 🌟 Features

### ✅ What Works
- **Two-way messaging** - Bot can receive and send messages
- **Command handling** - `/start`, `/help`, `/precise`, `/analyze`
- **Rich formatting** - Bold, italic, escaped text
- **Source citations** - Links to legal documents
- **Error handling** - Graceful error messages
- **Admin notifications** - Get notified when new users start the bot
- **Session tracking** - Track user queries and statistics

### 🆓 Why Telegram vs WhatsApp
- **Completely FREE** - No costs for messaging
- **Official API** - Reliable and well-documented
- **No browser automation** - Simple HTTP API
- **Rich features** - Commands, formatting, file sharing
- **No business verification** - Works immediately

## 🚀 Production Deployment

For production use, consider:

1. **Webhook mode** instead of polling for better performance
2. **SSL certificate** for webhook security
3. **Database integration** for user management
4. **Rate limiting** to prevent abuse
5. **Logging and monitoring** for reliability

## 📞 Support

If you encounter issues:

1. Check the logs in `logs/telegram_bot.log`
2. Ensure FastAPI backend is running and accessible
3. Verify your bot token and environment configuration
4. Test the bot with simple commands first

## 🎉 You're Done!

Your Telegram bot is now ready to provide legal advice to users in Albania. The bot uses the same AI backend as your web and Streamlit interfaces, ensuring consistent and accurate responses.

**Next Steps:**
- Share your bot username with users
- Monitor usage in the logs
- Consider adding more features like file upload support
- Set up production deployment with webhooks

---

*Need help? Check the main project documentation or create an issue in the repository.*
