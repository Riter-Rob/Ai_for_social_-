"""
Flask/crisis_response.py

Multilingual crisis response templates for Ethiopian users.
When the model detects a crisis, this module generates an empathetic,
culturally appropriate response with local Ethiopian helpline contacts.

Languages supported:
  am  — Amharic (አማርኛ)
  om  — Afan Oromo
  ti  — Tigrinya (ትግርኛ)
  en  — English
  mixed — Uses Amharic + English

Author: Robel Tesfaye (https://github.com/Riter-Rob)
"""

# ─── Ethiopian Crisis Helpline Numbers ────────────────────────────────────────
HELPLINES = {
    "health_hotline":      "8335",
    "amanuel_hospital":    "011 275 7680",
    "amanuel_hospital_2":  "011 275 7681",
    "red_cross":           "907",
    "police_emergency":    "911",
}


# ─── Response Templates ───────────────────────────────────────────────────────

def get_crisis_response(lang: str = "am", risk_level: str = "HIGH") -> dict:
    """
    Return a crisis response dict for the given language.

    Args:
        lang:       Language code: "am", "om", "ti", "en", "mixed"
        risk_level: "HIGH" or "MEDIUM"

    Returns:
        dict with:
          "message"  — main empathetic text
          "hotlines" — formatted hotline string
          "followup" — short follow-up encouragement line
          "lang"     — language used
    """
    templates = {
        "am": _amharic_response(risk_level),
        "om": _oromo_response(risk_level),
        "ti": _tigrinya_response(risk_level),
        "en": _english_response(risk_level),
        "mixed": _mixed_response(risk_level),
    }

    # Default to Amharic for unknown languages (most common in Ethiopia)
    return templates.get(lang, templates["am"])


def _amharic_response(risk_level: str) -> dict:
    if risk_level == "HIGH":
        message = (
            "ብቻህን/ሽ አይደለህም/ሽ። "
            "የምታልፍበትን ስሜት እንረዳለን — ሸክሙ ከባድ እንደሆነ ይሰማናል። "
            "አሁን ከዚህ ወዲያ ማዶ ዕርዳታ አለ። "
            "እባክህ/ሽ ዛሬ ወደ ሙያ ባለሙያ ደውሉ።"
        )
        followup = "ሕይወትህ/ሽ ዋጋ አለው። ዕርዳታ መጠየቅ ጥንካሬ ነው።"
    else:
        message = (
            "ሸክሙ ከባድ ሲሆን ማን ዘንድ ሄዶ ማነጋገር ጥሩ ነው። "
            "ብቻህን/ሽ ሆነህ/ሽ ማድረግ አለብህ/ሽ ብለህ/ሽ አታስብ/ቢ። "
            "ዕርዳታ ሁልጊዜ ይገኛል።"
        )
        followup = "ስሜትህ/ሽ ያስፈልጋናል። ስለ ሁኔታህ/ሽ ማውራት ይረዳ ይሆናል።"

    return {
        "message": message,
        "hotlines": (
            f"📞 የጤና ሆትላይን (ነፃ): {HELPLINES['health_hotline']}\n"
            f"🏥 አማኑኤል ሆስፒታል: {HELPLINES['amanuel_hospital']}\n"
            f"🚑 ቀይ መስቀል: {HELPLINES['red_cross']}"
        ),
        "followup": followup,
        "lang": "am",
    }


def _oromo_response(risk_level: str) -> dict:
    if risk_level == "HIGH":
        message = (
            "Kophaa miti. Yeroo rakkoo keessatti deeggarsi jira. "
            "Obbolessi/Obboleettii keenya, yaadni kee/tee nu barbaachisa. "
            "Karaa gargaarsa argachuu dandeessa."
        )
        followup = "Lubbu kee gatii qaba. Gargaarsa gaafachuun jabina."
    else:
        message = (
            "Yeroo ulfaataa keessa jirta. "
            "Nama amanamaa waliin haasa'uun si gargaaruu danda'a. "
            "Gargaarsi yeroo hunda argamu danda'a."
        )
        followup = "Nuti si waliin jirra. Hin yaadda'iin."

    return {
        "message": message,
        "hotlines": (
            f"📞 Health Hotline (free): {HELPLINES['health_hotline']}\n"
            f"🏥 Amanuel Hospital: {HELPLINES['amanuel_hospital']}\n"
            f"🚑 Red Cross: {HELPLINES['red_cross']}"
        ),
        "followup": followup,
        "lang": "om",
    }


def _tigrinya_response(risk_level: str) -> dict:
    if risk_level == "HIGH":
        message = (
            "ባዕልኻ/ኻ ኣይኮንካን/ኺ። "
            "ሓሳብካ/ኪ ኣጸቢቕና ንርዳእ — ሕሰምካ/ኪ ከቢድ ምዃኑ ንፈልጥ። "
            "ሎሚ ሓገዝ ምሕታት ጸጋ ዩ።"
        )
        followup = "ሂወትካ/ኪ ዋጋ ኣለዋ። ሓገዝ ምሕታት ጥንካረ ዩ።"
    else:
        message = (
            "ብዙሕ ዘሸግር ዘሎ ይሃልዋ። "
            "ምስ ሓደ ሰብ ምዝርራብ ክሕግዝ ይኽእል ዩ። "
            "ሓገዝ ኩሉ ጊዜ ኣሎ።"
        )
        followup = "ሓቢርና ኢና። ትጥዓም ዘሎ ሓቢርና ክንፈልጦ ኢና።"

    return {
        "message": message,
        "hotlines": (
            f"📞 Health Hotline (free): {HELPLINES['health_hotline']}\n"
            f"🏥 Amanuel Hospital: {HELPLINES['amanuel_hospital']}\n"
            f"🚑 Red Cross: {HELPLINES['red_cross']}"
        ),
        "followup": followup,
        "lang": "ti",
    }


def _english_response(risk_level: str) -> dict:
    if risk_level == "HIGH":
        message = (
            "You are not alone. "
            "What you are feeling right now is real — and it takes courage to reach out. "
            "Please talk to someone today. Help is available right now."
        )
        followup = "Your life matters. Asking for help is a sign of strength, not weakness."
    else:
        message = (
            "It sounds like things are difficult right now. "
            "Talking to someone you trust — or a professional — can really help. "
            "You don't have to go through this alone."
        )
        followup = "We're here with you. Take it one step at a time."

    return {
        "message": message,
        "hotlines": (
            f"📞 Health Hotline (free): {HELPLINES['health_hotline']}\n"
            f"🏥 Amanuel Mental Hospital: {HELPLINES['amanuel_hospital']}\n"
            f"🚑 Ethiopian Red Cross: {HELPLINES['red_cross']}"
        ),
        "followup": followup,
        "lang": "en",
    }


def _mixed_response(risk_level: str) -> dict:
    """Bilingual Amharic + English for mixed/code-switched text."""
    if risk_level == "HIGH":
        message = (
            "ብቻህ/ሽ አይደለህ/ሽም — You are not alone.\n"
            "ዕርዳታ አሁን አለ — Help is available right now.\n"
            "Please reach out today. እባክህ/ሽ ዛሬ ደውሉ።"
        )
        followup = "ሕይወትህ/ሽ ዋጋ አለው። Your life matters."
    else:
        message = (
            "Talking to someone can help — ማውራት ይረዳ ይሆናል.\n"
            "You don't have to face this alone — ብቻህ/ሽ ሆነህ/ሽ ማድረግ አያስፈልግም."
        )
        followup = "We are here — እኛ እዚህ ነን."

    return {
        "message": message,
        "hotlines": (
            f"📞 Health Hotline (free): {HELPLINES['health_hotline']}\n"
            f"🏥 Amanuel Hospital: {HELPLINES['amanuel_hospital']}\n"
            f"🚑 Red Cross: {HELPLINES['red_cross']}"
        ),
        "followup": followup,
        "lang": "mixed",
    }


def format_for_web(response: dict) -> dict:
    """Format crisis response dict for JSON API response."""
    return {
        "is_crisis": True,
        "message": response["message"],
        "hotlines_text": response["hotlines"],
        "hotlines": HELPLINES,
        "followup": response["followup"],
        "lang": response["lang"],
    }


def format_for_telegram(response: dict) -> str:
    """Format crisis response as a Telegram Markdown message."""
    return (
        f"⚠️ *ድጋፍ ያስፈልጋል / Support Needed*\n\n"
        f"{response['message']}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🆘 *ዕርዳታ ለማግኘት / Get Help Now:*\n"
        f"{response['hotlines']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💙 {response['followup']}"
    )
