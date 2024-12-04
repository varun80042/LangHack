import json
import os
import requests
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASETS_PATH = r"E:\miscE\ml\LLM_Hackathon\datasets\data"
DATASETS_MICROLABS_USA = os.path.join(DATASETS_PATH, "microlabs_usa")

URLS = {
    "Acetazolamide Extended-Release Capsules": "https://www.microlabsusa.com/products/acetazolamide-extended-release-capsules/",
    "Amlodipine Besylate and Olmesartan Medoxomil Tablets": "https://www.microlabsusa.com/products/amlodipine-besylate-and-olmesartan-medoxomil-tablets/",
    "Amoxicillin and Clavulanate Potassium for Oral Suspension, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-for-oral-suspension-usp/",
    "Amoxicillin and Clavulanate Potassium Tablets, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-tablets-usp/",
    "Amoxicillin Capsules, USP": "https://www.microlabsusa.com/products/amoxicillin-capsules-usp/",
    "Aspirin and Extended-Release Dipyridamole Capsules": "https://www.microlabsusa.com/products/aspirin-and-extended-release-dipyridamole-capsules/",
    "Atorvastatin Calcium Tablets": "https://www.microlabsusa.com/products/atorvastatin-calcium-tablets/",
    "Bimatoprost Ophthalmic Solution": "https://www.microlabsusa.com/products/bimatoprost-ophthalmic-solution/",
    "Celecoxib capsules": "https://www.microlabsusa.com/products/celecoxib-capsules/",
    "Chlordiazepoxide Hydrochloride and Clidinium Bromide Capsules, USP": "https://www.microlabsusa.com/products/chlordiazepoxide-hydrochloride-and-clidinium-bromide-capsules-usp/",
}

def setup_prescribing_info_urls(urls_map):
    updated_urls = {}

    for key, value in urls_map.items():
        logging.info(f"Processing URL for: {key}")
        updated_urls[key] = {
            "product_url": value,
        }
        try:
            data = requests.get(value)
            data.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(data.text, "html.parser")
            h2 = soup.findAll("h2")

            got = False
            for h2_item in h2:
                txt = h2_item.get_text()
                if txt and txt.strip().lower() == "prescribing information":
                    child_url = h2_item.findAll("a")
                    if child_url:
                        href = child_url[0].get("href")
                        updated_urls[key]["prescribing_info_url"] = href
                        logging.info(f"Found prescribing info URL for {key}: {href}")
                        html = requests.get(href)
                        html.raise_for_status()  # Raise an error for bad responses
                        prescribing_soup = BeautifulSoup(html.text, "html.parser")
                        updated_urls[key]["prescribing_soup"] = prescribing_soup
                        got = True
                if got:
                    break
        except requests.RequestException as e:
            logging.error(f"Error fetching data for {key}: {e}")

    return updated_urls

def find_elements_with_text(soup):
    elements_with_text = []
    for element in soup.find_all(True):
        if element.name not in ["script", "style"]:
            if element.string or element.get_text(strip=True):
                elements_with_text.append(element)

    for elem in elements_with_text:
        logging.debug(f"Tag: {elem.name}, Text: {elem.get_text(strip=True)}")

    return

def get_text_below_anchor_with_special_handling(a_tag):
    result_text = []
    for sibling in a_tag.find_next_siblings():
        if sibling.name == "div":
            childs = sibling.children
            for child in childs:
                if child.name is not None:
                    if child.name.lower() == "table":
                        table_content = []
                        rows = sibling.find_all('tr')
                        for row in rows:
                           
                            cells = [cell.get_text(strip=False) for cell in row.find_all(['td', 'th'])]
                            # Join cells with a single space separator
                            table_content.append(" ".join(cells))
                        result_text.append("\n".join(table_content))

                    elif child.name.lower() == "img":
                        # Process image content
                        img_src = child.get('src', 'No src attribute')
                        img_alt = child.get('alt', 'No alt text')
                        result_text.append(f"Image: [src={img_src}, alt={img_alt}]")
                    else:
                        result_text.append(sibling.get_text(strip=True))

    # Join and return the result
    return "\n".join(result_text)

def get_all_sections(soup):
    atags = soup.findAll("a")
    info = dict()

    for atag in atags:
        if atag:
            at = atag.get("id")
            if at and at.startswith("anch_dj_dj-dj"):
                txt = get_text_below_anchor_with_special_handling(atag)
                info[atag.get_text()] = txt
    return info

def process_prescribing_soup(name, soup):
    results = get_all_sections(soup)
    results["product_name"] = name
    logging.info(f"Processed prescribing soup for: {name}")
    return results

def create_dataset_file(pth, result):
    # Ensure the directory exists
    os.makedirs(pth, exist_ok=True)  # Create the directory if it doesn't exist
    fname = os.path.join(pth, result["product_name"] + ".json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    logging.info(f"Created dataset file: {fname}")

if __name__ == '__main__':
    # Check if the dataset directory exists
    if os.path.exists(DATASETS_MICROLABS_USA):
        logging.info(f"Directory already exists: {DATASETS_MICROLABS_USA}")
    else:
        os.makedirs(DATASETS_MICROLABS_USA)  # Create the directory if it doesn't exist
        logging.info(f"Created directory: {DATASETS_MICROLABS_USA}")

    logging.info("Starting the scraping process...")
    modified_urls = setup_prescribing_info_urls(URLS)
    
    for k, v in modified_urls.items():
        if "prescribing_soup" in v:
            results = process_prescribing_soup(k, v["prescribing_soup"])
            create_dataset_file(DATASETS_MICROLABS_USA, results)
        else:
            logging.warning(f"No prescribing soup found for {k}")

    logging.info("Scraping process completed.")
