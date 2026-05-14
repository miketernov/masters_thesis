import time
import pandas as pd
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"


def build_search_url(name, coords):
    return f"https://yandex.ru/maps/?text={quote(str(name))}&ll={coords}&z=17"


def build_org_url(org_id):
    return f"https://yandex.ru/maps/org/{org_id}"


def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    return driver


def extract_org_id(driver, target_name):
    items = driver.find_elements(By.CSS_SELECTOR, 'div[data-object="search-list-item"]')

    print("Найдено search-list-item:", len(items))

    target_name_norm = target_name.lower().strip().replace("ё", "е")

    for i, item in enumerate(items[:10]):
        org_id = item.get_attribute("data-id")
        text = item.text.lower().strip().replace("ё", "е")

        print(f"{i}: org_id={org_id}, text={text[:150].replace(chr(10), ' | ')}")

        if org_id and target_name_norm in text:
            print("Выбран по совпадению названия:", org_id)
            return org_id

    if items:
        fallback_id = items[0].get_attribute("data-id")
        print("Совпадение не найдено, берем первый result:", fallback_id)
        return fallback_id

    return None


def extract_chain_flag(driver):
    print("\n--- chain_flg ---")

    elems = driver.find_elements(By.XPATH, "//*[contains(text(), 'Все филиалы сети')]")

    print("Найдено элементов:", len(elems))

    return 1 if elems else 0


def get_chain_flag(driver, wait, name, coords):
    # 1. поиск
    search_url = build_search_url(name, coords)
    print("SEARCH:", search_url)

    driver.get(search_url)

    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(4)

    # 2. достаем org_id
    org_id = extract_org_id(driver, name)

    if not org_id:
        return None

    # 3. переходим в карточку
    org_url = build_org_url(org_id)
    print("ORG URL:", org_url)

    driver.get(org_url)

    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(4)

    # 4. ищем сеть
    return extract_chain_flag(driver)


def main():
    df = pd.read_csv(INPUT_FILE)

    if "chain_flg" not in df.columns:
        df["chain_flg"] = None

    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        for i, row in df.iterrows():
            name = row["rest_name"]
            coords = row["coordinates"]

            print(f"\n[{i}] {name}")

            try:
                chain_flg = get_chain_flag(driver, wait, name, coords)
                df.at[i, "chain_flg"] = chain_flg
                print("chain_flg =", chain_flg)

            except Exception as e:
                print("Ошибка:", e)
                df.at[i, "chain_flg"] = None

            if i % 20 == 0:
                df.to_csv(OUTPUT_FILE, index=False)

            time.sleep(2)

    finally:
        driver.quit()

    df.to_csv(OUTPUT_FILE, index=False)
    print("Готово")


if __name__ == "__main__":
    main()