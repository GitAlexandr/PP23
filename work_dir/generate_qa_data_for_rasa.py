"""
Скрипт для генерации конфигов в RASA

WORKING_DIR - рабочая директория, которая содержит папки
    data: для входных данных
    output: для сгенерированных по каждому датасету конфигураций
"""

from typing import List, Text, Optional
from pathlib import Path
import pandas as pd
import json
from transliterate import translit
import warnings
from itertools import chain
from config import *

warnings.filterwarnings('ignore')


class RasaDataGenerator:
    """
    Вход: xlsx файл вида
    
    intent: str | questions: list | answers: list
    """
    
    def __init__(
            self,
            output_dir = OUTPUT_DIR) -> None:
        
        """
        
        """
        self.output_dir = output_dir

    def generate_endpoints_file(self,
                                url,
                                output_dir: Path,
                                output_filename: Text = 'endpoints.yml'):
        with open(output_dir.joinpath(output_filename), "w") as fout:
            fout.write(f"""action_endpoint:
  url: "{url}"
""")


    def save_nlu_to_file(
            self,
            intent_name: Text = 'test_intent',
            intent_examples: List = ['раз', '2 (два)', 'три'],
            output_filename: Text = 'nlu.yml'):

        """
        Записывает данные для одного интента в указанный nlu.yml
        """

        output_filename = self.output_dir.joinpath(output_filename)
        with open(output_filename, 'a') as fout:
            intent_msgs_str = f"{TAB_SYMBOL * 2}- " + f'\n{TAB_SYMBOL * 2}- '.join(intent_examples.split('.'))

            fout.write(f"""- intent: {intent_name}
    {TAB_SYMBOL}examples: 
    {intent_msgs_str}

    """)


    def save_rules_to_file(
            self,
            intent_names: Text = ['test_intent', 'Два интент', 'Три интент'],
            output_filename: Text = "rules.yml"):
        """
        Записывает правила 
        вида intent_name -> utter_intent_name
        в указанный rules.yml
        """
        output_filename = self.output_dir.joinpath(output_filename)

        with open(output_filename, 'a') as fout:
            # fout.write("rules:\n")

            for intent_name in intent_names:
                fout.write(f"""- rule: new rule about {intent_name}
  steps:
  - intent: {intent_name}
  - action: utter_{intent_name}
""")
                
    def save_intent_and_action_names(
            self,
            intent_names: List[str],
            action_names = None,
            domain_output_filename="domain.yml"):
        """
        Записывает список интентов и список правил в файл domain.yml
        """
        domain_output_filename = self.output_dir.joinpath(domain_output_filename)

        separator = '\n- '
        intents_str = separator+separator.join(intent_names)
        if action_names is not None:
            actions_str = separator+separator.join(action_names)
        else:
            actions_str = separator+separator.join(f"utter_{intent_name}" for intent_name in intent_names)

        template = f"""

intents:{intents_str}

actions:{actions_str}

"""
        with open(domain_output_filename, 'a') as fout:
            fout.write(template)

    def save_intent_responses_to_file(
            self,
            intent_name: Text = 'test_intent',
            intent_response_examples: List = ['ответ 1', 'ответ 2 (два)', 'ответ три'],
            responses_output_filename: Text = "responses.yml"):
        """
        Записывает базу данных ответов в указанный файл домена (domain.yml)
        """

        output_filename = self.output_dir.joinpath(responses_output_filename)

        responses_str = f"{TAB_SYMBOL * 2}- " + f'\n{TAB_SYMBOL * 2}- text: '.join(intent_response_examples.split('.'))
        with open(output_filename, 'a') as fout:
            fout.write(f"""{TAB_SYMBOL}utter_{intent_name}:\n{responses_str}""")




    def __write_headers__(self,
                          nlu_output_filename='nlu.yml',
                          responses_output_filename='responses.yml',
                          rules_output_filename="rules.yml",
                          domain_output_filename='domain.yml',
                          config_output_filename='config.yml'
                          ):
        
        """
        Записывает метаданные во все конфигурационные файлы
        nlu, rules, responses, domain
        """
        nlu_output_filename, \
        responses_output_filename, \
        rules_output_filename, \
        domain_output_filename, \
        config_output_filename = [
                                self.output_dir.joinpath(filename)\
                                  for filename in [
                                              nlu_output_filename, \
                                                responses_output_filename, \
                                                rules_output_filename, \
                                                domain_output_filename, \
                                                config_output_filename]
                                  ]
        
        with open(nlu_output_filename, 'w') as nlu_fout, \
        open(responses_output_filename, 'w') as responses_fout, \
        open(rules_output_filename, 'w') as rules_fout, \
        open(domain_output_filename, 'w') as domain_fout, \
        open(config_output_filename, 'w') as config_fout:
            
            nlu_fout.write("""
version: '3.1'

nlu:

""")
            responses_fout.write("responses:\n\n")
            rules_fout.write("rules:\n\n")
            domain_fout.write("""version: '3.1'
session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: false
""")
            config_fout.write(f"""
language: ru


pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: CountVectorsFeaturizer
- name: DIETClassifier
  epochs: {DIET_EPOCHS}
  num_transformer_layers: 4
  transformer_size: 256
  use_masked_language_model: true
  drop_rate: 0.25
  weight_sparsity: 0.7
  batch_size: [64, 256]
  embedding_dimension: 100
  hidden_layer_sizes:
    text: [512, 128]
"""
                              )


    def run_pipeline_for_hierarchical_classifier(self,
                                                 df,
                                                 use_subintents = "single",
                                                 **filenames):

        self.output_dir.mkdir(parents=True, exist_ok=True) # create output_dir for dataset_name
        
        intent_names = df['intent'].unique().tolist()
        
        self.__write_headers__(**filenames)
        self.save_intent_and_action_names(intent_names=intent_names,
                                          domain_output_filename=filenames['domain_output_filename'])

        nlu_row_name = 'subintent' if use_subintents == "hierarchical" else 'intent'

        for _, row in df.iterrows():
            intent_examples = row['questions']
            
            self.save_nlu_to_file(intent_name=row[nlu_row_name],
                                  intent_examples=intent_examples,
                                  output_filename=filenames['nlu_output_filename']
                                  )
            self.save_intent_responses_to_file(intent_name=row[nlu_row_name],
                                               intent_response_examples=row['answers'],
                                               responses_output_filename=filenames['responses_output_filename'])
        self.save_rules_to_file(intent_names=intent_names,
                                output_filename=filenames['rules_output_filename'])
        if use_subintents == "hierarchical":
            self.update_config(intent_names=intent_names,
                               n_epochs_for_response_selector=100,
                               config_output_filename=filenames['config_output_filename'])
        if use_subintents == 'default':
            
            action_names = df['intent'].map(lambda x: f"utter_{x}").tolist()
            webhooks = [f"http://localhost:{port}/webhook" for port in (5005 + i for i in range(len(action_names)))]
            json_filename = 'webhooks_map.json'
            with open(self.output_dir.joinpath(json_filename), 'w') as fout:
                action_names_webhooks = dict(zip(action_names, webhooks))
                json.dump(action_names_webhooks,
                          fout,
                          ensure_ascii=False,
                          indent=4)
                
            generate_actions(output_dir=self.output_dir,
                             json_filename=json_filename,
                             expert_actions=action_names)
            for action_name, webhook_url in action_names_webhooks.items():
                expert_dir = self.output_dir.joinpath(action_name)
                expert_dir.mkdir(exist_ok=True)
                self.generate_endpoints_file(output_dir = expert_dir,
                                             output_filename=f"endpoints.yml",
                                             url=webhook_url)

    def update_config(self,
                      intent_names,
                      config_output_filename = 'config.yml',
                      n_epochs_for_response_selector = 10):
        template_for_single_intent = """- name: ResponseSelector
  epochs: {epochs}
  retrieval_intent: {intent_name}
  num_transformer_layers: 4
  transformer_size: 256
  use_masked_language_model: false
  drop_rate: 0.25
  weight_sparsity: 0.7
  batch_size: [64, 256]
  embedding_dimension: 30
  hidden_layer_sizes:
    text: [512, 128]
"""
        template = f"""
pipeline:
{''.join(template_for_single_intent.format(epochs = n_epochs_for_response_selector,
                                           intent_name=intent_name) for intent_name in intent_names)}"""
        config_output_filename = self.output_dir.joinpath(config_output_filename)
        with open(config_output_filename, 'w') as fout:
            fout.write(template)

def clean_and_transliterate(intent_name):
    def clean(string: str):
        string = string.lower().strip() if isinstance(string, str) else ''
        for symbol, replace_value in {
            ',': "",
            ".": "",
            '"': '',
            ' ': '_',
        }.items():
            string = string.replace(symbol, replace_value)
        return string

    res = translit(clean(intent_name), 'ru', reversed=True)
    return res



def transform_intent_names(df):
    
    """
    Переводит имя интента в валидную для RASA строку 
    """
    df['intent'] = df['intent'].apply(clean_and_transliterate)

    df['intent'] = df['intent'].fillna('').astype(str)
        
def split_df_by_intents(df):

    intent_names = df.intent.unique().tolist()


    intent_dataframes = {intent: df[df['intent'] == intent]
                        for intent in intent_names}
    return intent_dataframes

def create_subintents(intent_df, intent_name, use_subintents='hierarchical'):
    """true subintents == hierarchy of intents"""

    n_subintents = intent_df.shape[0]

    if use_subintents == 'default':
        create_intent_name = lambda x: "{intent_name}".format(intent_name=intent_name)
    else:
        sep = '/' if use_subintents == 'hierarchical' else '_'
        create_intent_name = lambda x: "{intent_name}{sep}question_{x}".format(intent_name=intent_name,
                                                                               sep=sep,
                                                                               x=x)

    intent_df['subintent'] = range(1, n_subintents + 1)
    intent_df['subintent'] = intent_df['subintent'].map(
        create_intent_name)

def merge_intent_dataframes(intent_dataframes):

    df_s = [intent_dataframe \
        for _, intent_dataframe in intent_dataframes.items()
        ]
        
    final_df = pd.concat(df_s, ignore_index=True)

    return final_df

def pipe(working_dir: Path, dataset_name: Text, use_subintents: bool  ):

    data_filename = working_dir.joinpath(f'{dataset_name}.xlsx')
    output_filenames = {
        f"{config_type}_output_filename": f'{config_type}_{dataset_name}.yml'
        for config_type in
        ('nlu', 'responses', 'rules', 'domain', 'config')
    }
    print(output_filenames)

    qa_df = pd.read_excel(data_filename)\
        [["intent", "questions_merged", "answers_merged"]]
    qa_df.fillna("", inplace=True)

    qa_df.rename(columns={
        "questions_merged": "questions",
        "answers_merged": "answers"
    },
    inplace = True)



    for col_name in ['answers', 'questions']:
        qa_df[col_name] = qa_df[col_name]
    
    transform_intent_names(qa_df)
    intent_dataframes = split_df_by_intents(qa_df)
    
    
    for intent, intent_dataframe in intent_dataframes.items():
        if use_subintents == 'default':
            intent_dict = {
                col: [list(chain.from_iterable(intent_dataframe[col].tolist()))]
                for col in ('questions', 'answers')
            }
            intent_dataframes[intent] = pd.DataFrame.from_dict(data=intent_dict)
        create_subintents(intent_dataframes[intent], intent, use_subintents=use_subintents)

    qa_df = merge_intent_dataframes(intent_dataframes)
    if use_subintents != "hierarchical":
        qa_df['intent'] = qa_df['subintent']


    config_generator = RasaDataGenerator(working_dir.joinpath('output', dataset_name))
    config_generator.run_pipeline_for_hierarchical_classifier(df = qa_df, use_subintents=use_subintents, **output_filenames)


"""
Необходимые колонки
`intent`, `questions_merged`: List, `answers_merged`: List
"""
if __name__ == "__main__":

    dataset_names = ["merge"]
    use_subintents = ['default', 'single'][-1] #True
    for dataset_name in dataset_names[:1]:
        print(f"Processing {dataset_name}")
        pipe(working_dir=WORKING_DIR, dataset_name=dataset_name, use_subintents = use_subintents)
        print("OK")
