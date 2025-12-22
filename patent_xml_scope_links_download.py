import time, os, csv
from selenium.webdriver.edge.options import Options
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re



def xml_sub(url):
    edge_options = Options()
    edge_options.use_chromium = True
    edge_options.add_argument("--headless==new")
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(8)
    wait = WebDriverWait(driver, 20)
    element = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//a[@href='#detailMainForm:MyTabViewId:NATIONALDOCUMENTS']"))
    )
    element.click()
    # driver.find_element('xpath', "//a[@href='#detailMainForm:MyTabViewId:NATIONALDOCUMENTS']").click()
    time.sleep(4)
    zip_links = driver.find_elements('xpath',
                                     "//table//a[contains(@href, '.zip') and contains(@href, 'filename')]")
    time.sleep(2)
    for link in zip_links:
        link.click()
        time.sleep(10)
    return True

def select_appropriate_link(links):
    """Selects the appropriate link based on filename suffix availability."""
    # Extract filenames and map suffixes to links
    suffix_map = {}
    for link in links:
        match = re.search(r'filename=([^&]+)\.zip', link)
        if match:
            filename = match.group(1)
            suffix = filename[-2:]  # Get last 2 characters
            suffix_map[suffix] = link

    # Decision tree based on availability
    if all(suffix in suffix_map for suffix in ['A1', 'A2', 'B1', 'B2']):
        return suffix_map['B2']
    elif all(suffix in suffix_map for suffix in ['A1', 'A2', 'B1']):
        return suffix_map['B1']
    elif all(suffix in suffix_map for suffix in ['A1', 'A2']):
        return suffix_map['A1']

    return None


def get_download(url):
    edge_options = Options()
    edge_options.use_chromium = True
    edge_options.add_experimental_option("prefs", {
        "download.default_directory": "/home/sim-ind-0041/Documents/Patent_downloads/CO2_methanation",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    # edge_options.add_argument("--headless==new")
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(5)
    wait = WebDriverWait(driver, 20)
    try:
        element = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[@href='#detailMainForm:MyTabViewId:NATIONALDOCUMENTS']"))
        )
        element.click()
        # driver.find_element('xpath', "//a[@href='#detailMainForm:MyTabViewId:NATIONALDOCUMENTS']").click()
        time.sleep(4)
        zip_links = driver.find_elements('xpath', "//table//a[contains(@href, '.zip') and contains(@href, 'filename')]")
        time.sleep(2)
        link_B2 = link_B1 = link_A1 = None
        for link in zip_links:
            href = link.get_attribute("href")
            match = re.search(r'filename=([^&]+)\.zip', href)
            if match:
                filename = match.group(1)
                if filename.endswith('B2'):
                    link_B2 = link
                elif filename.endswith('B1'):
                    link_B1 = link
                elif filename.endswith('A1'):
                    link_A1 = link
        if link_B2:
            link_B2.click()
            time.sleep(10)
        elif link_B1:
            link_B1.click()
            time.sleep(10)
        elif link_A1:
            link_A1.click()
            time.sleep(10)

        driver.close()
        return True
    except:
        driver.close()
        return False

def main():
    csv_file = '/home/sim-ind-0041/Documents/my_server_files/patent_urls_methanation.csv'
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            url = row['urls']  # Adjust 'url' to match your CSV column name
            status = get_download(url)
            print(status, url)
            



if __name__ == "__main__":
    main()


