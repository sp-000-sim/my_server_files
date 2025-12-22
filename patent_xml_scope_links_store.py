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
    # edge_options.add_argument("--headless==new")
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


def get_xml(url, patent_num):

    edge_options = Options()
    edge_options.use_chromium = True
    edge_options.add_experimental_option("prefs", {
        "download.default_directory": "/home/simreka/Downloads/download_tmp",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    # edge_options.add_argument("--headless==new")
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(5)
    driver.find_element('xpath', '//*[@id="advancedSearchForm:advancedSearchInput:input"]').send_keys(f"LGP:EN and IC_EX:{patent_num}")
    time.sleep(4)
    # driver.find_element('xpath', '//*[@id="advancedSearchForm:searchOfficesOption"]/div/div[1]/button').click()
    # time.sleep(2)
    # driver.find_element('xpath', '/html/body/div[2]/div[5]/div/div[1]/form/div[3]/div[1]/div[1]/div/div/span[1]/div/div[2]/div/div[5]/div[1]/input').click()
    # time.sleep(2)
    driver.find_element('xpath', '//*[@id="advancedSearchForm:searchButton"]').click()
    time.sleep(8)
    # try:
    wait = WebDriverWait(driver, 20)

    driver.find_element('xpath',
                        '/html/body/div[2]/div[4]/div/div[1]/div[2]/div/form[1]/div/div[1]/div[2]/div/select[1]/option[4]').click()
    time.sleep(5)
    href_final = []
    for i in range(1, 10):
        print(i)
        try:
            if i == 2:
                driver.find_element('xpath', '/html/body/div[2]/div[4]/div/div[1]/div[2]/div/form[1]/div/div[2]/a').click()
            elif i > 2:
                driver.find_element('xpath',
                                    '/html/body/div[2]/div[4]/div/div[1]/div[2]/div/form[1]/div/div[2]/a[2]').click()

            time.sleep(7)
            table = driver.find_element("xpath", f"//*[@id='resultListForm:resultTable']/div/table")
            time.sleep(3)
            hrefs = []
            link_elements = table.find_elements(By.XPATH, ".//a[@target='_self']")
            time.sleep(3)
            for element in link_elements:
                href = element.get_attribute("href")
                if href:
                    hrefs.append(href)
            href_final.extend(hrefs)
        except:
            print("Exception occurred at:", i)
            break

    return href_final

def main():
    class_id = '(C10G47 OR C10G65 OR B01J21/12 OR B01J23/88) OR FP:(hydrocracking OR hydrocracker OR "zeolite catalyst")'
    url = 'https://patentscope.wipo.int/search/en/advancedSearch.jsf'
    href_final = get_xml(url=url, patent_num=class_id)
    print(len(href_final))
    df = pd.DataFrame()
    df["urls"] = href_final
    df.to_csv(f"patent_urls_hydrocracking.csv")



if __name__ == "__main__":
    main()


