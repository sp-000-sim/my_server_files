import time
# from cgi import print_environ_usage

import pandas as pd
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_pdf(url, class_num):
    edge_options = Options()
    edge_options.use_chromium = True
    # edge_options.add_argument("--headless")
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(6)
    driver.find_element('xpath', '//*[@id="structuredSearchForm:conditions:2:conditionField:input"]/option[32]').click()
    time.sleep(2)
    driver.find_element('xpath', '//*[@id="structuredSearchForm:conditions:2:conditionValue:input"]').send_keys(class_num)
    time.sleep(2)
    driver.find_element('xpath', '//*[@id="structuredSearchForm:commandStructuredSearch"]/span[1]').click()
    time.sleep(5)
    driver.find_element('xpath', '/html/body/div[2]/div[4]/div/div[1]/div[2]/div/form[1]/div/div[1]/div[2]/div/select[1]/option[4]').click()
    time.sleep(5)
    href_final = []
    for i in range(1, 701):
        print(i)
        try:
            if i != 1:
                driver.find_element('xpath', '/html/body/div[2]/div[4]/div/div[1]/div[2]/div/form[1]/div/div[2]/a').click()
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

# def related_docs(patent_url):
#     response = requests.get(patent_url)
#     print(response)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     # print(soup)
#     # patent_family_members = soup.find_all('div', class_='ps-field ps-biblio-field ')
#     patent_family_members = soup.find_all('div', class_='patent-family-member')
#     print(patent_family_members)
#     href_values = []
#     for member in patent_family_members:
#         href_values.append(member['href'])
#     return href_values

def related_docs(patent_url):
    edge_options = Options()
    edge_options.use_chromium = True
    # edge_options.add_argument("--headless")
    driver = webdriver.Edge(options=edge_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(4)
    # WebDriverWait(driver, 10).until(
    #     EC.presence_of_all_elements_located(By.XPATH, '//div[@id="sdetailMainForm:MyTabViewId:j_idt8393:j_idt9142"]//a[contains(@class="patent-family-member")')
    # )
    links = driver.find_element(By.XPATH, '//*[@class="ps-field--value ps-biblio-field--value"]')
    # links = driver.find_elements(By.XPATH, '//div[@id="sdetailMainForm:MyTabViewId:j_idt8393:j_idt9142"]//a[contains(@class, "patent-family-member")')
    time.sleep(4)
    print('links', links)
    for link in links:
        print(link.get_attribute('href'))
    href_values = []
    return href_values

url = "https://patentscope.wipo.int/search/en/structuredSearch.jsf"
classification_number = "IC_EX:(C07C1/12 OR B01J23/72 OR B01J23/755) OR FP:(methanation OR \"CO2 hydrogenation\" OR Sabatier) AND LGP:EN"
pat_url_collection = []
patent_urls = get_pdf(url=url, class_num=classification_number)
print("Number of patents for C07B: ", len(patent_urls))
df = pd.DataFrame()
df["urls"] = patent_urls
df.to_csv(f"patent_links_methanation.csv")
# related_url = related_docs(patent_urls)


