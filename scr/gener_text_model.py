# from YaGPT import YaGPT, YaGPTException

# # Replace with your actual values
# folder_id = "b1gjocqmnjl20h2d2528"
# iam_token = "y0_AgAAAABE5ACHAATuwQAAAADe37C6dCOE7pouSmerwnJ8iQXmdIUjbZ8"

# # Create a LanguageModel instance
# lm = YaGPT(folder_id, iam_token)

# try:
#     result = lm.instruct(
#             model="general",
#             instruction_text="Найди ошибки в тексте и исправь их",
#             request_text="Ламинат подойдет для укладке на кухне или в детской комнате",
#             max_tokens=1500,
#             temperature=0.6)

#     if result:
#         for alternative in result:
#             print(f"Generated Text: {alternative['text']}")
# except YaGPTException as e:
#     print(f"Language Model Error: {e}")

# from selenium import webdriver
# from selenium.webdriver.common.by import By

# driver = webdriver.Chrome()

# driver.get("https://www.phind.com/")

# title = driver.title

# driver.implicitly_wait(0.5)

# text_box = driver.find_element(by=By.XPATH, value='//*[@id="__next"]/div/main/div/div/div/div[1]/div[2]/div/form/div[1]/textarea')
# submit_button = driver.find_element(by=By.XPATH, value='//*[@id="__next"]/div/main/div/div/div/div[1]/div[2]/div/form/div[1]/div[2]/button[2]')

# text_box.send_keys("Hi")
# submit_button.click()

# message = driver.find_element(by=By.XPATH, value='//*[@id="__next"]/div/main/div/div[2]/div[4]')
# text = message.text

# driver.quit()


																					