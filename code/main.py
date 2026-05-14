import re
from datetime import datetime, timezone
from telethon import TelegramClient
import pandas as pd

api_id = 123
api_hash = "xxx"

client = TelegramClient("session_name", api_id, api_hash)

START_DATE = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

STATUS_PATTERNS = {
    "closed": [
        r"#закрытие\b",
    ],
}

def extract_status(text: str):
    if not text:
        return None

    text = text.lower()

    for status, patterns in STATUS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return status

    return None

def extract_district(text: str):
    if not text:
        return None

    text = text.lower()

    # Ищем район после символа |
    m = re.search(r"\|\s*#([^\s#]+)", text)
    return m.group(1) if m else None

async def main():
    rows = []
    channel = await client.get_entity("raidedrests")

    async for msg in client.iter_messages(channel, offset_date=END_DATE):
        if not msg.date:
            continue

        if msg.date < START_DATE:
            break

        text = msg.message or ""
        status = extract_status(text)

        # В датасет попадают только посты с #открытие или #закрытие
        if status is None:
            continue

        rows.append({
            "message_id": msg.id,
            "date": msg.date.isoformat(),
            "status": status,
            "district": extract_district(text),
            "text": text,
        })

    df = pd.DataFrame(rows)
    df.to_csv("telegram_2026_03_01_13_filtered.csv", index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows")

with client:
    client.loop.run_until_complete(main())