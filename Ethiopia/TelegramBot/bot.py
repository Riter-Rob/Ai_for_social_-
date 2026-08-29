"""
Ethiopia/TelegramBot/bot.py

Ethiopian Multilingual Mental Health Telegram Bot.

Supports Amharic, Afan Oromo, Tigrinya, English, and mixed text.
Detects crisis/suicidal ideation and responds with:
  - Empathetic messages in the user's language
  - Ethiopian crisis helpline numbers (8335, Amanuel Hospital, Red Cross)

Commands:
  /start      — Welcome message
  /help       — How to use the bot
  /resources  — List all Ethiopian mental health resources
  /about      — About this project

Prerequisites:
  pip install python-telegram-bot python-dotenv

Author: Robel Tesfaye (https://github.com/Riter-Rob)
"""

import sys
import os
import logging

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from Ethiopia.TelegramBot.config import (
    TELEGRAM_BOT_TOKEN,
    CRISIS_THRESHOLD,
    MAX_TEXT_LENGTH,
    DEBUG_MODE,
    BOT_NAME,
)
from Ethiopia.Model.inference import EthiopianCrisisDetector
from Flask.crisis_response import (
    get_crisis_response,
    format_for_telegram,
    HELPLINES,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Load detector once at startup ───────────────────────────────────────────
logger.info("Loading Ethiopian Crisis Detector...")
detector = EthiopianCrisisDetector(threshold=CRISIS_THRESHOLD)
logger.info("Detector ready.")


# ════════════════════════════════════════════════════════════════════════════
#  Safe (non-crisis) response templates
# ════════════════════════════════════════════════════════════════════════════

SAFE_RESPONSES = {
    "am": (
        "✅ መልእክትህ/ሽ ደህና ይመስላል።\n\n"
        "ስሜቶችህ/ሽ ይቀጥሉ ማውራቱ ደህና ነው — "
        "ሁልጊዜ ማዳመጥ አለ። 💙\n\n"
        "_ምንም ጊዜ ሸክምህ/ሽ ቢከብድ — /resources ይጻፉ ለዕርዳታ_"
    ),
    "om": (
        "✅ Ergaan kee nagaa fakkaata.\n\n"
        "Yaada keessaa haasa'uun ni gargaara — "
        "yeroo hunda dhaggeeffanaa jirra. 💙\n\n"
        "_Gargaarsa yoo barbaadde /resources barreessi_"
    ),
    "ti": (
        "✅ መልእኽቲኻ/ኺ ጽቡቕ ይመስል።\n\n"
        "ዝስምዓካ/ኺ ምዝርራብ ሓጋዚ ዩ — "
        "ኩሉ ጊዜ ምስ ኢና። 💙\n\n"
        "_ሓገዝ እንተ ደሊኻ/ኺ /resources ጸሓፍ_"
    ),
    "en": (
        "✅ Your message seems okay.\n\n"
        "Talking about how you feel is always a good step — "
        "we're here to listen. 💙\n\n"
        "_If you ever need support, type /resources_"
    ),
    "mixed": (
        "✅ Your message seems okay / መልእክትህ ደህና ይመስላል.\n\n"
        "We are here / እኛ እዚህ ነን 💙\n\n"
        "_Type /resources for support links / ዕርዳታ ለማግኘት /resources ይጻፉ_"
    ),
}

UNCERTAIN_RESPONSES = {
    "am": (
        "🤔 ስሜቶችህ/ሽ ሸክም ሊሆን ቢቻል — ማናቃቶቹ አናውቅም ።\n\n"
        "ቢፈልጉ ስለ ሁኔታህ/ሽ ተጨማሪ ማውራት ይቻላል። "
        "ምንም ጊዜ ዕርዳታ ይገኛል።"
    ),
    "en": (
        "🤔 It sounds like things might be difficult.\n\n"
        "If you'd like to share more, I'm here to listen. "
        "Help is always available — type /resources anytime."
    ),
}


# ════════════════════════════════════════════════════════════════════════════
#  Command Handlers
# ════════════════════════════════════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command — Welcome message in Amharic + English."""
    user = update.effective_user
    name = user.first_name if user.first_name else "ወዳጄ"

    welcome = (
        f"👋 *እንኳን ደህና መጣህ/ሽ, {name}!*\n\n"
        f"🤖 እኔ *{BOT_NAME}* ነኝ — "
        f"ቦቴ ነኝ ስሜቶችህ/ሽ ሰምቶ ዕርዳታ ለማቅረብ የሚሞክር።\n\n"
        f"*Welcome, {name}!*\n"
        f"I am the *{BOT_NAME}* — a mental health support bot "
        f"for Ethiopian users. I can understand Amharic, Afan Oromo, "
        f"Tigrinya, and English.\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📝 *ምን ማድረግ ይቻላል / What I can do:*\n"
        f"• ስሜቶችህ/ሽ ተናገር — ሰማ / Share how you feel — I'll listen\n"
        f"• ፍቺ ቀናት ካጋጠሙ — ዕርዳታ ላፈልግ / If you're struggling — I'll find help\n"
        f"• ለአደጋ ሁኔታ ቁጥሮች / Crisis hotline numbers\n\n"
        f"❗ *ፍቺ ቀናት ካሉ / In crisis? Type:* /resources\n\n"
        f"_ለምን ጀመርህ/ሽ? ስሜቶችህ/ሽ ን ጻፍ/ጽፊ_ 👇"
    )

    await update.message.reply_text(welcome, parse_mode=constants.ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "🆘 *እንዴት መጠቀም ይቻላል / How to use this bot:*\n\n"
        "1. ስሜቶችህ/ሽ ን ጻፍ/ጽፊ (ማንኛቸውም ቋንቋ) / "
        "Just type how you feel in any language\n"
        "2. ቦቱ ያዳምጣል እና ምላሽ ይሰጣል / The bot listens and responds\n"
        "3. ፍቺ ወቅቶች ሲኖሩ — ዕርዳታ ያቀርባል / "
        "If crisis signs are detected, it provides help\n\n"
        "*Commands:*\n"
        "/start — ጀምር / Start over\n"
        "/help — ይህ ዝርዝር / This list\n"
        "/resources — ሁሉ ዕርዳታ ቁጥሮች / All crisis resources\n"
        "/about — ስለ ቦቱ / About this bot\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *ማሳሰቢያ / Disclaimer:*\n"
        "_ይህ ቦቱ ዶክተር ወይም ሕክምና አይደለም። "
        "ፍቺ ቀናት ሲኖሩ ሙያ ባለሙያ ያነጋግሩ።_\n"
        "_This bot is not a doctor. Always consult a professional in crisis._"
    )
    await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)


async def resources(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /resources command — All Ethiopian mental health resources."""
    resources_text = (
        "🆘 *የኢትዮጵያ አደጋ ሕክምና ቁጥሮች / Ethiopian Crisis Helplines*\n\n"
        f"📞 *የጤና ሆትላይን (ነፃ) / Health Hotline (Free):*\n"
        f"   📲 `{HELPLINES['health_hotline']}`\n\n"
        f"🏥 *አማኑኤል አዕምሮ ሆስፒታል / Amanuel Mental Hospital:*\n"
        f"   📲 `{HELPLINES['amanuel_hospital']}`\n"
        f"   📲 `{HELPLINES['amanuel_hospital_2']}`\n"
        f"   📍 አዲስ አበባ / Addis Ababa\n\n"
        f"🚑 *ቀይ መስቀል / Ethiopian Red Cross:*\n"
        f"   📲 `{HELPLINES['red_cross']}`\n\n"
        f"🚔 *ፖሊስ ድንገተኛ / Police Emergency:*\n"
        f"   📲 `{HELPLINES['police_emergency']}`\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ *ሌሎች ዕርዳታ / Other Support:*\n"
        f"• አዲስ አበባ ዩኒቨርሲቲ ካውንሴሊንግ / AAU Counseling Center\n"
        f"• ዩኒቨርሲቲ Counseling Directorate (ሁሉ ዩኒቨርሲቲዎች)\n"
        f"• Mental Health Ethiopia (NGO)\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💙 _ብቻህ/ሽ አይደለህ/ሽም — You are not alone._"
    )
    await update.message.reply_text(resources_text, parse_mode=constants.ParseMode.MARKDOWN)


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about command."""
    about_text = (
        f"🤖 *{BOT_NAME}*\n\n"
        f"ይህ ቦቱ AI ቴክኖሎጂ ተጠቅሞ ስሜቶችህ/ሽ ን ይሰማ — "
        f"ፍቺ ቀናት ሲኖሩ ዕርዳታ ለማቅረብ ይሞክራል።\n\n"
        f"This bot uses AI to understand how you're feeling and "
        f"connect you to mental health support in Ethiopia.\n\n"
        f"*Supported Languages:*\n"
        f"🇪🇹 Amharic (አማርኛ)\n"
        f"🇪🇹 Afan Oromo (Afaan Oromoo)\n"
        f"🇪🇹 Tigrinya (ትግርኛ)\n"
        f"🌍 English\n\n"
        f"*Developer:* Robel Tesfaye\n"
        f"*GitHub:* github.com/Riter-Rob/Ai\\_for\\_social\\_-\n\n"
        f"_MIT Licensed — Built for Ethiopia 🇪🇹_"
    )
    await update.message.reply_text(about_text, parse_mode=constants.ParseMode.MARKDOWN)


# ════════════════════════════════════════════════════════════════════════════
#  Message Handler — Core logic
# ════════════════════════════════════════════════════════════════════════════

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle every incoming text message.
    1. Run Ethiopian crisis detection
    2. Respond based on risk level
    """
    text = update.message.text
    if not text or not text.strip():
        return

    # Truncate very long messages
    text = text[:MAX_TEXT_LENGTH]

    if DEBUG_MODE:
        user = update.effective_user
        logger.debug(f"Message from {user.id} ({user.username}): {text[:80]}")

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        result = detector.predict(text)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        await update.message.reply_text(
            "⚠️ ቴክኒካዊ ስህተት ተከሰተ። / Technical error occurred. Please try again."
        )
        return

    lang       = result["lang"]
    risk_level = result["risk_level"]
    label      = result["label"]

    logger.info(f"Prediction: label={label}, risk={risk_level}, lang={lang}, conf={result['confidence']:.2f}")

    # ── Respond based on risk ───────────────────────────────────────────────
    if label == 1:
        # CRISIS or MEDIUM risk → empathetic response + hotlines
        crisis_data = get_crisis_response(lang=lang, risk_level=risk_level)
        telegram_msg = format_for_telegram(crisis_data)

        await update.message.reply_text(
            telegram_msg,
            parse_mode=constants.ParseMode.MARKDOWN,
        )

    else:
        # SAFE → warm, encouraging response
        response_template = SAFE_RESPONSES.get(lang, SAFE_RESPONSES["en"])
        await update.message.reply_text(
            response_template,
            parse_mode=constants.ParseMode.MARKDOWN,
        )


# ════════════════════════════════════════════════════════════════════════════
#  Error handler
# ════════════════════════════════════════════════════════════════════════════

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Update {update} caused error: {context.error}")


# ════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Start the bot."""
    print(f"Starting {BOT_NAME}...")
    print(f"Crisis threshold: {CRISIS_THRESHOLD}")
    print(f"Debug mode: {DEBUG_MODE}\n")

    # Build application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start",     start))
    application.add_handler(CommandHandler("help",      help_command))
    application.add_handler(CommandHandler("resources", resources))
    application.add_handler(CommandHandler("about",     about))

    # Register message handler (all text messages)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Register error handler
    application.add_error_handler(error_handler)

    # Start polling
    print("Bot is running. Press Ctrl+C to stop.\n")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
