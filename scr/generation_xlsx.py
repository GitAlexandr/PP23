import openpyxl

# Create a workbook and select the active sheet
wb = openpyxl.Workbook()
sheet = wb.active

# Define the headers
headers = ['main_theme', 'question', 'questions_paraphrase', 'answer', 'answer_summary', 'intent', 'answer_merge', 'question_merge']

# Write the headers to the first row
for i, header in enumerate(headers, start=1):
    sheet.cell(row=1, column=i, value=header)

# Save the workbook
wb.save('output.xlsx')