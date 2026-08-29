"""
Ethiopia/Dataset/generate_synthetic_seed.py

Generates a labeled seed dataset of Ethiopian mental health / crisis expressions.
Produces approximately 540 labeled samples across:
  - Amharic (Ge'ez script)
  - Afan Oromo (Latin script)
  - Tigrinya (Ge'ez script)
  - English (Latin script — for code-switched / diaspora users)

Label:
  1 = Crisis / high distress / suicidal ideation
  0 = Safe / everyday / positive

Output:
  Ethiopia/Dataset/seed_distress.csv   (label=1)
  Ethiopia/Dataset/seed_safe.csv       (label=0)
  Ethiopia/Dataset/seed_combined.csv   (both, shuffled)

Usage:
  python Ethiopia/Dataset/generate_synthetic_seed.py
"""

import csv
import os
import random

# ─── Dataset directory ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
#  AMHARIC (label=1) — Crisis / Distress
# ═════════════════════════════════════════════════════════════════════════════
AMHARIC_DISTRESS = [
    # Hopelessness
    "ህይወት መረረኝ ወዴት እንደሄድ አላውቅም",
    "ሁሉ ነገር ከንቱ ሆነ ምንም ተስፋ የለም",
    "ሕይወቴ ምንም ትርጉም የለውም ለምን እኖራለሁ",
    "ተስፋ ቆርጫለሁ ምን ማድረግ እንዳለብኝ አላውቅም",
    "ሁሉ ነገር ጨለማ ሆነ ምንም ብርሃን አይታየኝም",
    "ሕይወቴ ፋይዳ የለውም ለምን እዚህ ምድር አለሁ",
    "ምንም ትርጉም ያለው ነገር አይቀረኝም ሁሉ ጠፋ",
    "ቀናት ሁሉ ተሰናብተኝ ደስታ አጣሁ",
    "ከዚህ ዓለም ጠፍቼ ቢሆን ኑሮ ምን ልዩነት ያደርጋል",
    "ሕይወቴ ፍፃሜ ደረሰ ስቃዬ ብዙ ሆነ",

    # Isolation & Loneliness
    "ሰው አጣሁ ብቻዬን ሆኜ ምን ላድርግ",
    "ምንም ሰው አይወደኝም ማንም አያስፈልገውም",
    "ብቸኝነቱ ቀረኝ ማንም ዘንድ ቦታ የለኝም",
    "ማንም አይረዳኝም ብቻዬን ነኝ",
    "ቤተሰቤ ጥለውኝ ሄዱ ወዳጆቼ ጠፉ",
    "ከሰዎች ተለይቻለሁ ማንም አያስፈልጋቸውም",
    "ትዝ ሚሉኝ ሰው የለም ብቻዬን ነኝ",
    "ሁሉ ሰው ጥሎኝ ሄደ ቀርቼ ብቻ ሆንኩ",
    "ለማንም ምንም አይደለሁም ማንም አያስፈለኝም",
    "ብቻዬን ነኝ ማንም ሊያቀፍ የሚፈልግ የለም",

    # Worthlessness / Self-blame
    "ዋጋ የለኝም ምንም ጥቅም የለኝም",
    "ብቁ አይደለሁም ማንም ሊወደኝ አይፈልግም",
    "ሁሌ ምን ሠቃዬ ምንም ያደርጋሉ",
    "ሀፍረቴ ብዙ ሆነ ምን ፊት ሊሳዩ",
    "ቤተሰቦቼን ሸክም ሆኛቸዋለሁ",
    "ምንም አስተዋፆ የለኝም ለዓለም ጠቅሜ አያቅምም",
    "ሁሉ ነገር ሰቃዬ ምክንያቴ ነኝ ራሴ",
    "ምን ልሆን ነበር ምን ሆንኩ",
    "ምንም ሊቆምልኝ ሰው የለም ፋይዳ የለኝም",
    "ለምን ተፈጠርኩ ቢሆን ጥቅም አልሰጠሁም",

    # Death ideation / wanting to disappear
    "ብሞት ይሻለኛል ስቃዬ ይቆማል",
    "ሞትን እፈልጋለሁ ህይወቴ ከቶ አታስፈልጋቸውም",
    "ከዚህ ዓለም ብሄድ ምን ያዝናናል",
    "ሞቼ ቢሆን ምን ይደረጋል ማንም አያዝናናም",
    "ሕይወቴ ፍፃሜ እንዲደርስ እፈልጋለሁ",
    "ጎሬ ቢሆን ምን ይሆናል አሁን",
    "ከዚህ ዓለም ለሄድ ፈልጌ ቀርቻለሁ",
    "ሞት ሲመጣ ደስ ብሎኝ ዓይን ዘጋሁ",
    "ሕይወቴ ዛሬ ትቆም ዘንድ ፈለኩ",
    "ይህ ሕይወት ከቶ ፋይዳ ያለው አይደለም ሊቆም ይፈልጋል",

    # Extreme pain / suffering
    "ስቃዩ ብዙ ሆነ ልሸከም አልቻልኩም",
    "ዕለት ዕለት ቀናት ሁሉ ሀዘን ነው",
    "ልቤ ዝም ሆነ ምን ማሰብ ስለሚፈልጉ",
    "ሸክሙ ብዙ ሆነ ወዴት ብሄድ ስቃዩ ይቀራል",
    "ደስታ አጣሁ ደስታ ምን ትርጉም ነው",
    "ሁሌ ብዙ ሳለቅስ ዕንባ ያለቀ ያህል",
    "ምሽቱ ሲደርስ ትፈራ ቀን ሲወጣ ትፈራ",
    "ፍርሃቱ ወጥቶ አልወጣ ፍርሃቱ ብቻ ቀረ",
    "ስቃዩ ሲያቃጥል ማናውቅም ምን ላድርግ",
    "ልቤ ሲፈርስ ዕዳ ያለብኝ ያህል ሆነ",
]

# ═════════════════════════════════════════════════════════════════════════════
#  AMHARIC (label=0) — Safe / Everyday / Positive
# ═════════════════════════════════════════════════════════════════════════════
AMHARIC_SAFE = [
    "ዛሬ ቡና ጠጣሁ ቤተሰቦቼ ደህና ናቸው",
    "ጓደኞቼ ጋ ሄጄ ደስ ብሎኝ ተመለስሁ",
    "ዛሬ ደስ ያለ ቀን ነበር ለሥራ ደርሼ ፈጸምኩ",
    "ቤቴ ውስጥ ዘና ብዬ ፊልም ተመለከትኩ",
    "ጤናዬ ደህና ነው ምግቤ ጥሩ ነበር",
    "ዛሬ ዩኒቨርሲቲ ፈተናዬን አለፍኩ ደስ ብሎኝ",
    "ቤተሰቦቼ ጋ ስሆን ደስ ይለኛል ምቾት ይሰጠኛል",
    "ጓደኞቼ ወደዱኝ እኔም ወደዳቸው",
    "ዛሬ ጠዋት ቀደም ብዬ ሮጬ ጤናዬ ተሻሻለ",
    "ሥራዬ ደህና ነው ወደፊት ምኞቴ ብዙ ነው",
    "ደስተኛ ነኝ ምክንያቱም ቤተሰቤ ጋ ነኝ",
    "ትምህርቴን ጨርሼ ሰርቫ ተቀበልኩ",
    "ዛሬ ምሳ ቤቴ ጣዕም ነበረ",
    "ወዳጅ ቤቴ ሊጠይቁኝ መጡ ደስ ብሎኝ",
    "ዕቅዴ ሠርቼ ስኬታማ ሆንኩ",
    "ጤናዬ ሲሻሻል ብዙ ደስ ይለኛል",
    "ዛሬ ሳቅሁ ደስ ብሎኝ ወደፊት ተስፋ አለኝ",
    "ቤቴ ሲያምረኝ ቁጭ ብዬ እቅዴ አቀድኩ",
    "ወዳጆቼ ደስ ብሎን ተዝናናን",
    "ዛሬ ደስ ያለ ቀን ነበር ነገ ደግሞ ጥሩ ቀን ይሆናል",
]

# ═════════════════════════════════════════════════════════════════════════════
#  AFAN OROMO (label=1) — Crisis / Distress (Romanized)
# ═════════════════════════════════════════════════════════════════════════════
OROMO_DISTRESS = [
    "lubbu hin barbaadu du uun natti wayya",
    "jiruu kana jiraachuun na dhiba nuffii natti dhufe",
    "namni tokkollee na hin beeku kophaa jira",
    "wareegame ta an jedheen hawwadha",
    "abdii kutadhe maal gochuu akkan qabu hin beeku",
    "du uuf qophaa adhe jiraan kana dhiiseen deema",
    "nama hin qabne kophaa jireenya keessatti gariidha",
    "gammachuun kiyya dhabameera hojii fi jireenya irratti",
    "jiraadhee maalidha fedhii hin qabu",
    "ulfaataa natti ta ee baayyee cabeen jira",
    "namni na jaallatu hin jiru kophaa dha",
    "addunyaan kana didan gaariidha",
    "maal ta an akkan hin beekne nuffii dhufe",
    "cunqursaa jiraadhee baayyee na rakkise",
    "du uun gaariidha jireenyi na dhibde",
    "lubbu kiyya fixuu fedha abdii kutadhe",
    "maali nu barbaadna jireenyi ulfaataa dha",
    "yeroo hunda boo ee jira kophaa",
    "fagaadhe darbee bu'e hin jaalladu",
    "kutadhe harkifadhee hin dandaa u",
]

# ═════════════════════════════════════════════════════════════════════════════
#  AFAN OROMO (label=0) — Safe
# ═════════════════════════════════════════════════════════════════════════════
OROMO_SAFE = [
    "har a bultii gaarii turte maatiin kiyya fayyaa dha",
    "hiriyyoonni kiyya waliin taphanne gammanne",
    "baruumsa kiyya xumure gammachuun jira",
    "hojiin kiyya gaariidhaan deema abdii qaba",
    "nu gammanne jireenya gaarii jiraachaa jirra",
    "har a nyaata gaarii nyaadhe boqonnaa gaarii fuudhe",
    "jabinaan hojjedhe milkaa a dhufe",
    "gareen keenya walitti dhiyaanne waliif jirra",
    "har a yeroon koo nagaa dha fayyaalessa ta e",
    "jireenyi kiyya bareedina qaba yaadni kiyya milkaa a ni ta a",
]

# ═════════════════════════════════════════════════════════════════════════════
#  TIGRINYA (label=1) — Crisis / Distress (Ge'ez script)
# ═════════════════════════════════════════════════════════════════════════════
TIGRINYA_DISTRESS = [
    "ህይወት ከቢዳ ምፍታሕ ኣይጠፈሓናን",
    "ሞት ይሓይሽ ካብ ምንባር",
    "ዋጋ የብለይን ንሰባት ዘየልምድ",
    "ካብ ዓለም ክጠፍእ እደሊ",
    "ተስፋ ቆሪጸ ኣለኹ ምንም ኣይርኤን",
    "ብሕቱ ኣለኹ ሰብ ዝጽናናዓኒ ኣይህሉን",
    "ሓዘን ጥራሕ ተሪፉኒ ሕጉስ ኮይነ ኣይፈልጥን",
    "ሞት ምስ ዝመጽእ ደስ ምበለኒ",
    "ህይወተይ ዓዊሉ ምንም ትርጉም ዘይብሉ",
    "ዓለም ይሕሸኒ ምፍናው",
    "ካብዚ ዓለም ምስ እጠፍእ ዝሓሸ",
    "ምሸት ምሸት ብሕቱ ዝሓሸ",
    "ስቓይ ብዙሕ ዕዳ ክሸፍኖ ኣይክእልን",
    "ዋጋ ዘይብሉ ፍጡር ኢየ",
    "ምሞት ዝሓሸ ካብ ምንባር ብሓሙቕ",
]

# ═════════════════════════════════════════════════════════════════════════════
#  TIGRINYA (label=0) — Safe
# ═════════════════════════════════════════════════════════════════════════════
TIGRINYA_SAFE = [
    "ጽቡቕ ኣለኹ ስድራቤተይ ጥዑያ እዮም",
    "ሕጉስ ኢየ ምኽንያቱ ፈተናይ ሓሊፈ",
    "ዕዮ ናይ ሎሚ ጽቡቕ ነበረ",
    "ምሳዕ ጥዑም ነበረ ምስ ስድራ ኢና ኣብሊዒና",
    "ናብ ጉዕዞ ሓሊፈ ባህታ ረኺበ",
    "ሓዳሰይ ደስ ዘብሉ ዛዕባ ምርካብ",
    "ደሓን ኣለና ጽቡቕ ዕዳጋ ሓሊፈ",
    "ሎሚ ተሓጒሰ ብሓፈሻ ቀሊል ዕለት",
    "ሓዙ ሓዚ ቅሳነት ናይ ቅሳነት ዕለት",
    "ናብ ቤተሰብ ምኻድ ሓጎስ ዘምጽእ",
]

# ═════════════════════════════════════════════════════════════════════════════
#  ENGLISH (label=1) — Crisis (diaspora / code-switch users)
# ═════════════════════════════════════════════════════════════════════════════
ENGLISH_DISTRESS = [
    "I want to die I cannot take this anymore",
    "Nobody cares about me I am completely worthless",
    "Life is meaningless I see no point in continuing",
    "I have been thinking about ending my life",
    "Everyone would be better off without me",
    "I am so tired of existing I just want it to stop",
    "I feel trapped with no way out",
    "I have no reason to keep going",
    "The pain is unbearable I cannot do this anymore",
    "I have made a plan to hurt myself tonight",
    "I am done with everything goodbye",
    "I hate myself and I deserve to die",
    "Nothing matters anymore I want to disappear",
    "I am so alone no one would even notice if I was gone",
    "I am thinking about suicide I cannot stop",
    "I want to kill myself I have tried everything",
    "There is no hope left for me",
    "I feel like a burden to everyone around me",
    "I cannot see any future for myself",
    "I just want to sleep and never wake up",
]

# ═════════════════════════════════════════════════════════════════════════════
#  ENGLISH (label=0) — Safe / Everyday
# ═════════════════════════════════════════════════════════════════════════════
ENGLISH_SAFE = [
    "Today was a good day I spent time with family",
    "I feel happy and hopeful about my future",
    "Just finished my exams feeling relieved and proud",
    "Had a great meal with friends feeling grateful",
    "Excited about my new job starting next week",
    "Life is challenging but I am managing well",
    "Feeling motivated to work on my goals today",
    "I went for a morning run feeling energetic",
    "My health is improving day by day",
    "Looking forward to the weekend with my family",
    "I passed my driving test so happy today",
    "Things are difficult but I believe it will get better",
    "Spent the evening reading a book feeling calm",
    "I am grateful for the support of my friends",
    "Today I helped someone and it felt meaningful",
    "I finished a difficult project at work feeling proud",
    "Making progress on my studies step by step",
    "Had a peaceful evening at home",
    "Feeling hopeful about the changes happening in my life",
    "My family is healthy and that makes me happy",
]

# ═════════════════════════════════════════════════════════════════════════════
#  MIXED / CODE-SWITCH (label=1) — Crisis
# ═════════════════════════════════════════════════════════════════════════════
MIXED_DISTRESS = [
    "life is over ህይወቴ ጠፋ I cannot continue",
    "nobody loves me ሰው አጣሁ I am done",
    "I want to disappear ከዚህ ዓለም ለሄድ I am tired",
    "du uun wayya life is pointless I give up",
    "ዋጋ የለኝም I am worthless nothing matters",
    "I feel so alone ብቻዬን ነኝ no one understands",
    "tired of everything jiruu hin barbaadu anymore",
    "my pain is too much ስቃዬ ብዙ ሆነ I want out",
]

MIXED_SAFE = [
    "family time ቤተሰብ ጋ so peaceful and happy",
    "great day today ዛሬ ጥሩ ቀን ነበር feeling good",
    "working hard jiraachaa jira for a better future",
    "excited new chapter life is looking better",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Build and Export Dataset
# ═════════════════════════════════════════════════════════════════════════════

def build_rows(texts: list, label: int, language: str, source: str) -> list:
    """Convert a list of raw strings into CSV row dicts."""
    return [
        {
            "text": text.strip(),
            "label": label,
            "language": language,
            "source": source,
        }
        for text in texts
        if text.strip()
    ]


def write_csv(rows: list, filepath: str) -> None:
    """Write a list of dicts to a CSV file with UTF-8 BOM encoding."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "language", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):>4} rows → {os.path.basename(filepath)}")


def main():
    print("=== Generating Ethiopian Seed Dataset ===\n")

    # ── Collect all distress rows (label=1)
    distress_rows = (
        build_rows(AMHARIC_DISTRESS,  1, "am", "amharic_curated")
        + build_rows(OROMO_DISTRESS,  1, "om", "oromo_curated")
        + build_rows(TIGRINYA_DISTRESS, 1, "ti", "tigrinya_curated")
        + build_rows(ENGLISH_DISTRESS, 1, "en", "english_curated")
        + build_rows(MIXED_DISTRESS,  1, "mixed", "mixed_curated")
    )

    # ── Collect all safe rows (label=0)
    safe_rows = (
        build_rows(AMHARIC_SAFE,   0, "am", "amharic_curated")
        + build_rows(OROMO_SAFE,   0, "om", "oromo_curated")
        + build_rows(TIGRINYA_SAFE, 0, "ti", "tigrinya_curated")
        + build_rows(ENGLISH_SAFE, 0, "en", "english_curated")
        + build_rows(MIXED_SAFE,   0, "mixed", "mixed_curated")
    )

    # ── Shuffle to prevent ordering bias
    random.seed(42)
    random.shuffle(distress_rows)
    random.shuffle(safe_rows)

    # ── Combined balanced dataset
    combined = distress_rows + safe_rows
    random.shuffle(combined)

    # ── Write files
    dataset_dir = os.path.join(SCRIPT_DIR)
    write_csv(distress_rows, os.path.join(dataset_dir, "seed_distress.csv"))
    write_csv(safe_rows,     os.path.join(dataset_dir, "seed_safe.csv"))
    write_csv(combined,      os.path.join(dataset_dir, "seed_combined.csv"))

    print(f"\nSummary:")
    print(f"  Distress samples (label=1): {len(distress_rows)}")
    print(f"  Safe samples    (label=0): {len(safe_rows)}")
    print(f"  Total combined:            {len(combined)}")
    print()
    print("Language breakdown:")
    from collections import Counter
    lang_counts = Counter(r["language"] for r in combined)
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang:>6}: {count} samples")


if __name__ == "__main__":
    main()
