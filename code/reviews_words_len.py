import re
import pandas as pd

INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"ё", "е", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    reviews = pd.read_csv(INPUT_FILE)

    # нормализация текста
    reviews["review_text_norm"] = reviews["review_text"].apply(normalize_text)

    # длина в символах
    reviews["review_len_chars"] = reviews["review_text_norm"].apply(len)

    # длина в словах
    reviews["review_len_words"] = reviews["review_text_norm"].apply(
        lambda x: len(x.split())
    )

    # сохраняем
    reviews.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Готово: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()