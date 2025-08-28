#!/usr/bin/env python3
"""
Quick script to get your Telegram Chat ID
Run this after you've messaged your bot
"""

import requests
import json

def get_chat_id():
    print("🤖 Telegram Chat ID Finder")
    print("=" * 40)
    
    bot_token = input("Enter your TELEGRAM_BOT_TOKEN: ").strip()
    
    if not bot_token:
        print("❌ Bot token is required!")
        return
    
    print("\n📝 Instructions:")
    print("1. Go to your bot on Telegram")
    print("2. Send any message to your bot (e.g., 'Hello')")
    print("3. Press Enter here to continue...")
    input()
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['ok'] and data['result']:
                print("\n✅ Found messages:")
                print("=" * 40)
                
                for update in data['result']:
                    if 'message' in update:
                        chat = update['message']['chat']
                        from_user = update['message'].get('from', {})
                        
                        print(f"Chat ID: {chat['id']}")
                        print(f"Chat Type: {chat['type']}")
                        print(f"User Name: {from_user.get('first_name', '')} {from_user.get('last_name', '')}")
                        print(f"Username: @{from_user.get('username', 'N/A')}")
                        print(f"Message: {update['message'].get('text', 'N/A')}")
                        print("-" * 40)
                
                # Get the most recent chat ID
                latest_update = data['result'][-1]
                if 'message' in latest_update:
                    chat_id = latest_update['message']['chat']['id']
                    print(f"\n🎯 Your CHAT ID is: {chat_id}")
                    print(f"\nAdd this to your .env file:")
                    print(f"TELEGRAM_ADMIN_CHAT_ID={chat_id}")
                
            else:
                print("❌ No messages found!")
                print("Make sure you've sent a message to your bot first.")
        
        else:
            print(f"❌ Error: {response.status_code}")
            print("Check your bot token is correct.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_chat_id()
