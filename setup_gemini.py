#!/usr/bin/env python3
"""
Setup script for Google Gemini API integration
Helps users get and configure their free Gemini API key
"""

import os
import sys
from dotenv import load_dotenv, set_key

def setup_gemini_api():
    """Setup Google Gemini API key"""
    
    print("🚀 SETTING UP GOOGLE GEMINI 2.0 FLASH (FREE)")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        print("📝 Creating .env file from template...")
        
        # Copy from .env.example if it exists
        if os.path.exists(".env.example"):
            with open(".env.example", 'r') as f:
                example_content = f.read()
            with open(".env", 'w') as f:
                f.write(example_content)
            print("✅ Created .env file from template")
        else:
            # Create basic .env file
            with open(".env", 'w') as f:
                f.write("""# Albanian Legal RAG System - Environment Variables

# Google Gemini API Configuration (FREE)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Search Configuration  
SIMILARITY_THRESHOLD=0.15
MAX_CONTEXT_LENGTH=4000
""")
            print("✅ Created basic .env file")
    
    # Load environment variables
    load_dotenv()
    
    # Check current Gemini API key
    current_key = os.getenv("GEMINI_API_KEY")
    
    if current_key and current_key != "your_gemini_api_key_here":
        print(f"✅ Gemini API key already configured: {current_key[:20]}...")
        
        # Test the API key
        try:
            import google.generativeai as genai
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content("Test")
            print("✅ API key is working correctly!")
            return True
        except Exception as e:
            print(f"❌ API key test failed: {e}")
            print("Let's set up a new API key...")
    
    print()
    print("🔑 GET YOUR FREE GEMINI API KEY:")
    print("1. Visit: https://aistudio.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy your API key")
    print()
    
    # Get API key from user
    while True:
        api_key = input("🔤 Paste your Gemini API key here: ").strip()
        
        if not api_key:
            print("❌ Please enter a valid API key")
            continue
        
        if api_key == "your_gemini_api_key_here":
            print("❌ Please enter your actual API key, not the placeholder")
            continue
        
        # Test the API key
        print("🧪 Testing API key...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content("Hello, test message")
            
            if response.text:
                print("✅ API key is working!")
                
                # Save to .env file
                set_key(".env", "GEMINI_API_KEY", api_key)
                print("💾 API key saved to .env file")
                break
            else:
                print("❌ API key test failed - no response")
                
        except Exception as e:
            print(f"❌ API key test failed: {e}")
            
        retry = input("🔄 Try again? (y/n): ").lower()
        if retry != 'y':
            return False
    
    print()
    print("🎉 GEMINI SETUP COMPLETE!")
    print("✅ Your Albanian Legal RAG system is now powered by Google Gemini 2.0 Flash")
    print("🚀 Run: streamlit run app.py")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("google.generativeai", "google-generativeai"),
        ("streamlit", "streamlit"), 
        ("sentence_transformers", "sentence-transformers"),
        ("dotenv", "python-dotenv")
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main setup function"""
    print("🏛️ ALBANIAN LEGAL RAG - GEMINI SETUP")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return False
    
    print()
    
    # Setup Gemini API
    if setup_gemini_api():
        print()
        print("🎯 NEXT STEPS:")
        print("1. Run: python demo.py (to test the system)")
        print("2. Run: streamlit run app.py (to start the web interface)")
        print()
        print("💡 Benefits of Gemini 2.0 Flash:")
        print("   • Completely FREE to use")
        print("   • Very fast response times")
        print("   • Excellent Albanian language support")
        print("   • Large context window")
        print("   • Latest Google AI technology")
        
        return True
    else:
        print("❌ Setup failed. Please try again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)
