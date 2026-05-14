import re
import pandas as pd
from transformers import pipeline

# =========================
# 1. ПУТИ К ФАЙЛАМ
# =========================

INPUT_FILE = r"xxx"
OUTPUT_REVIEWS_FILE = r"xxx"

# =========================
# 2. КОЛОНКИ
# =========================

REVIEW_TEXT_COL = "review_text"
REVIEW_TEXT_NORM_COL = "review_text_norm"

# =========================
# 3. ПАТТЕРНЫ ЖАЛОБ
# =========================

SERVICE_PATTERNS = [
    r"\bхам\w*",
    r"\bгруб\w*",
    r"\bневежлив\w*",
    r"\bнеприветлив\w*",
    r"\bневоспитан\w*",
    r"\bбестактн\w*",
    r"\bвысокомерн\w*",
    r"\bнадменн\w*",
    r"\bпренебрежительн\w*",
    r"\bнеуважительн\w*",
    r"\bгрубиян\w*",
    r"\bнахамил\w*",
    r"\bнагрубил\w*",
    r"\bхамоват\w*",
    r"\bрезк\w* тон\w*",
    r"\bорал\w* на",
    r"\bнакричал\w*",
    r"\bкричал\w* на",
    r"\bигнор\w*",
    r"\bпроигнор\w*",
    r"\bбезразлич\w*",
    r"\bне внимател\w*",
    r"\bне замечал\w*",
    r"\bне подходил\w*",
    r"\bне смотрел\w* в сторон\w*",
    r"\bмимо прошел\w*",
    r"\bне реагировал\w*",
    r"\bожидани\w*",
    r"\bдолг\w*",
    r"\bмедлен\w*",
    r"\bне дожд\w*",
    r"\bждал\w* час\w*",
    r"\bждал\w* полчас\w*",
    r"\bждал\w* вечност\w*",
    r"\bпришлось ждать\w*",
    r"\bвремя ожидания\w*",
    r"\bзаставил\w* ждать\w*",
    r"\bтри\w* час\w* жда\w*",
    r"\bзабы\w*",
    r"\bотмен\w*",
    r"\bперепутал\w*",
    r"\bне то принес\w*",
    r"\bне тот заказ\w*",
    r"\bперепутали заказ\w*",
    r"\bне принял\w*",
    r"\bне обслуж\w*",
    r"\bне принесли меню\w*",
    r"\bне предложил\w*",
    r"\bне предупредил\w*",
    r"\bне объяснил\w*",
    r"\bне подош\w*",
    r"\bне встретил\w*",
    r"\bне проводил\w*",
    r"\bне поздоровал\w*",
    r"\bне улыбнул\w*",
    r"\bне помог\w*",
    r"\bотказал\w* помоч\w*",
    r"\bне извинил\w*",
    r"\bне принес\w* извинени\w*",
    r"\bне предложил\w* замен\w*",
    r"\bобсчитал\w*",
    r"\bобманул\w*",
    r"\bлишнее в счет\w*",
    r"\bне дал\w* сдач\w*",
    r"\bне принес\w* сдач\w*",
    r"\bошибка в счет\w*",
    r"\bотношени\w*",
    r"\bотвратител\w* сервис",
    r"\bужасн\w* сервис",
    r"\bкошмар\w* сервис",
    r"\bужасн\w* обслуживани\w*",
    r"\bотвратительн\w* обслуживани\w*",
    r"\bкошмарн\w* обслуживани\w*",
    r"\bсервис.*отвратит\w*",
    r"\bсервис.*ужас\w*",
    r"\bперсонал.*хам",
    r"\bперсонал.*груб",
    r"\bперсонал.*равнодуш\w*",
    r"\bофициант.*хам",
    r"\bофициант.*груб",
    r"\bофициант.*игнор\w*",
    r"\bкассир.*груб\w*",
    r"\bадминистратор.*груб\w*",
    r"\bбольше не (приду|вернус\w*|пойд\w*)",
    r"\bникому не (советую|рекомендую)",
    r"\bне рекомендую\b",
]

FOOD_PATTERNS = [
    # Вкус
    r"\bневкус\w*",
    r"\bбезвкус\w*",
    r"\bпресн\w*",
    r"\bгорьк\w*",
    r"\bкисл\w*",
    r"\bстранн\w* вкус\w*",
    r"\bнеприятн\w* вкус\w*",
    r"\bпротивн\w* вкус\w*",
    r"\bпривкус\w*",
    r"\bесть невозможн\w*",
    r"\bнесъедобн\w*",
    r"\bхолодн\w*",
    r"\bгорел\w*",
    r"\bподгорел\w*",
    r"\bпережар\w*",
    r"\bпережаренн\w*",
    r"\bне прожар\w*",
    r"\bнепрожаренн\w*",
    r"\bнедожар\w*",
    r"\bнедовар\w*",
    r"\bразварен\w*",
    r"\bраскисш\w*",
    r"\bрезинов\w*",
    r"\bсух\w*",
    r"\bпересолен\w*",
    r"\bслишком остр\w*",
    r"\bочень остр\w*",
    r"\bсыр(ой|ая|ое|ые|овато|оват)\w*",
    r"\bнесвеж\w*",
    r"\bне свеж\w*",
    r"\bпрокис\w*",
    r"\bиспорч\w*",
    r"\bпротух\w*",
    r"\bтухл\w*",
    r"\bтухлятин\w*",
    r"\bзалежал\w*",
    r"\bс душком\b",
    r"\bплохо пахн\w*",
    r"\bтухло пахн\w*",
    r"\bвоня\w*",
    r"\bнеприятн\w* запах\w*",
    r"\bзапах\w* тухл\w*",
    r"\bзапах\w* кисл\w*",
    r"\bволос\w* в\b",
    r"\bволосок\w* в\b",
    r"\bнашел\w* волос\w*",
    r"\bтаракан\w*",
    r"\bнасекомо\w*",
    r"\bмуха\w* в\b",
    r"\bплесен\w*",
    r"\bплесневел\w*",
    r"\bразогрет\w*",
    r"\bмикроволновк\w*",
    r"\bполуфабрикат\w*",
    r"\bразморожен\w*",
    r"\bзаморожен\w*",
    r"\bхимоз\w*",
    r"\bхимическ\w* вкус\w*",
    r"\bмаленьк\w* порци\w*",
    r"\bмаленьк\w* кусоч\w*",
    r"\bмикроскопическ\w* порци\w*",
    r"\bпорци\w* мизерн\w*",
    r"\bслишком жирн\w*",
    r"\bжирн\w*",
    r"\bочень жирн\w*",
    r"\bотрав\w*",
    r"\bпищевое отравлени\w*",
    r"\bстало плохо после\w*",
    r"\bзаболел\w* после\w*",
    r"\bеда.*ужас",
    r"\bеда.*кошмар",
    r"\bеда.*отврат",
    r"\bеда.*несъедобн\w*",
    r"\bкухня.*ужас\w*",
    r"\bкухня.*отврат\w*",
    r"\bблюдо.*холодн\w*",
    r"\bблюдо.*невкусн\w*",
]

PRICE_PATTERNS = [
    r"\bдорог\w*",
    r"\bдороговато\b",
    r"\bслишком дорог\w*",
    r"\bочень дорог\w*",
    r"\bдороже чем\b",
    r"\bвысок\w* цен\w*",
    r"\bзавышен\w*",
    r"\bнеоправдан\w*",
    r"\bнеадекватн\w* цен\w*",
    r"\bнесоразмерн\w*",
    r"\bнесоответстви\w* цен\w*",
    r"\bцена не соответствует\w*",
    r"\bсоотношени\w* цен\w*.*качеств\w*",
    r"\bбешен\w* цен\w*",
    r"\bбешен\w* деньг\w*",
    r"\bастрономическ\w* цен\w*",
    r"\bценник.*заоблачн\w*",
    r"\bценник.*безумн\w*",
    r"\bзаоблачн\w* цен\w*",
    r"\bконский ценник",
    r"\bконск\w* цен\w*",
    r"\bцены кусают\w*",
    r"\bцены кусачи\w*",
    r"\bграбеж\w*",
    r"\bграбит\w*",
    r"\bразвод\w* на деньг\w*",
    r"\bза что плат\w*",
    r"\bстолько денег за\b",
    r"\bтакая цена за\b",
    r"\bне стоит\b",
    r"\bне стоит своих денег\b",
    r"\bза такие деньги\b",
    r"\bсвоих денег\b",
    r"\bв других местах дешевле\b",
    r"\bнакрутк\w*",
    r"\bнаценк\w*",
    r"\bпереплат\w*",
    r"\bобсчитал\w*",
    r"\bлишнее в счет\w*",
    r"\bчек.*лишн\w*",
    r"\bв счете.*лишн\w*",
    r"\bдвойное списани\w*",
    r"\bошибка в счет\w*",
    r"\bцена.*ужас",
    r"\bцена.*кошмар",
    r"\bцена.*отврат\w*",
    r"\bцены.*ужас\w*",
    r"\bценник.*ужас\w*",
    r"\bбольш\w* цен\w*",
    r"\bне понимаю за что плат\w*",
]

# =========================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def clean_text(text):
    text = safe_text(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pick_text(row):
    norm_text = safe_text(row.get(REVIEW_TEXT_NORM_COL, ""))
    raw_text = safe_text(row.get(REVIEW_TEXT_COL, ""))

    if norm_text:
        return clean_text(norm_text)
    return clean_text(raw_text)

def contains_any_pattern(text, patterns):
    if not text:
        return 0
    for pattern in patterns:
        if re.search(pattern, text):
            return 1
    return 0

# =========================
# 5. ЧТЕНИЕ CSV
# =========================

try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, encoding="cp1251")

print(f"Загружено строк: {len(df)}")

# =========================
# 6. ПОДГОТОВКА ТЕКСТА
# =========================

df["text_for_nlp"] = df.apply(pick_text, axis=1)
df["has_text"] = df["text_for_nlp"].apply(lambda x: 1 if x else 0)

# =========================
# 7. SENTIMENT MODEL
# =========================

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
)

def map_stars_to_sentiment(label):
    # Примеры label: '1 star', '2 stars', ..., '5 stars'
    stars = int(label[0])

    if stars <= 2:
        return "negative", -1
    elif stars == 3:
        return "neutral", 0
    else:
        return "positive", 1

def predict_sentiment_batch(texts, batch_size=32):
    sentiment_labels = []
    sentiment_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_pipe(batch, truncation=True)

        for res in results:
            label, score_num = map_stars_to_sentiment(res["label"])
            sentiment_labels.append(label)
            sentiment_scores.append(score_num)

        print(f"Обработано отзывов: {min(i + batch_size, len(texts))} / {len(texts)}")

    return sentiment_labels, sentiment_scores

texts = df["text_for_nlp"].tolist()
sent_labels, sent_scores = predict_sentiment_batch(texts)

df["sentiment_label"] = sent_labels
df["sentiment_score"] = sent_scores

# =========================
# 8. ТЕМАТИЧЕСКИЕ ФЛАГИ
# =========================

df["service_complaint"] = df["text_for_nlp"].apply(
    lambda x: contains_any_pattern(x, SERVICE_PATTERNS)
)

df["food_complaint"] = df["text_for_nlp"].apply(
    lambda x: contains_any_pattern(x, FOOD_PATTERNS)
)

df["price_complaint"] = df["text_for_nlp"].apply(
    lambda x: contains_any_pattern(x, PRICE_PATTERNS)
)

# =========================
# 9. БИНАРНЫЕ SENTIMENT-ФЛАГИ
# =========================

df["is_negative"] = (df["sentiment_label"] == "negative").astype(int)
df["is_neutral"] = (df["sentiment_label"] == "neutral").astype(int)
df["is_positive"] = (df["sentiment_label"] == "positive").astype(int)

# =========================
# 10. НЕГАТИВ + ТЕМА
# =========================

df["negative_service"] = (
    (df["is_negative"] == 1) & (df["service_complaint"] == 1)
).astype(int)

df["negative_food"] = (
    (df["is_negative"] == 1) & (df["food_complaint"] == 1)
).astype(int)

df["negative_price"] = (
    (df["is_negative"] == 1) & (df["price_complaint"] == 1)
).astype(int)

# =========================
# 11. СОХРАНЕНИЕ
# =========================

df.to_csv(OUTPUT_REVIEWS_FILE, index=False, encoding="utf-8-sig")

print("Готово.")
print(f"Файл сохранен: {OUTPUT_REVIEWS_FILE}")
print("Новые колонки:")
print([
    "text_for_nlp",
    "has_text",
    "sentiment_label",
    "sentiment_score",
    "service_complaint",
    "food_complaint",
    "price_complaint",
    "is_negative",
    "is_neutral",
    "is_positive",
    "negative_service",
    "negative_food",
    "negative_price"
])