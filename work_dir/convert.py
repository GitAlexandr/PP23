import os
import pandas as pd

def read_xlsx(file_path):
    df = pd.read_excel(file_path)

    required_columns = ['intent', 'answers_merged', 'questions_merged']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
        return

    df['intent'] = df['intent'].astype(str)
    df['intent_type'] = df['intent'].apply(lambda x: 'questions_merged' if x.startswith('question_') else 'answers_merged')

    data = {group: df[df['intent_type'] == group][['intent', 'answers_merged', 'questions_merged']].to_dict(orient='records') for group in ['questions_merged', 'answers_merged']}
    return data

def write_yaml(data, output_file='output.yml'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry_type, entries in data.items():
            file.write(f"{entry_type}:\n")
            for entry in entries:
                file.write(f"  - intent: {entry['intent']}\n")
                file.write(f"    questions_merged: {entry['questions_merged']}\n")

def write_response_yaml(data, output_file='output.yml'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry_type, entries in data.items():
            file.write(f"{entry_type}:\n")
            for entry in entries:
                file.write(f"  - intent: {entry['intent']}\n")
                file.write(f"    answers_merged: {entry['answers_merged']}\n")
                res = entry['answers_merged']
                res = res.replace('[','')
                res = res.replace(']','')
                import json

                print(eval(res))
                

if __name__ == "__main__":
    file_path = 'data/merge.xlsx'
    nlu_output_file = 'output/merge/nlu_merge.yml'
    response_output_file = 'output/merge/response_merge.yml'

    result = read_xlsx(file_path)
    write_yaml(result, nlu_output_file)
    print(f"Данные записаны в файл: {nlu_output_file}")

    # Запись в файл response_merge.yml
    response_data = {'answers_merged': result['answers_merged']}
    write_response_yaml(response_data, response_output_file)
    print(f"Данные записаны в файл: {response_output_file}")
