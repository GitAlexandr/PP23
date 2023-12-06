import pandas as pd

class YourRasaConfigGenerator:
    def __init__(self):
        pass

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
            self.output_dir.joinpath(filename) \
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


    def __write_nlu(self, df, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for _, row in df.iterrows():
                intent_name = row['intent']
                questions = row['question_merge']
                file.write(f"## intent:{intent_name}\n")
                for question in questions:
                    file.write(f"- {question}\n")

    def __write_responses(self, df, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for _, row in df.iterrows():
                intent_name = row['intent']
                answers = row['answer_merge']
                for answer in answers:
                    file.write(f"{intent_name}:\n  - {answer}\n")

    def __write_rules(self, df, filename):
        # Добавьте вашу логику для создания правил (если есть)

    def __write_domain(self, df, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("intents:\n")
            for intent_name in df['intent'].unique():
                file.write(f"  - {intent_name}\n")
            file.write("\nresponses:\n")
            for _, row in df.iterrows():
                intent_name = row['intent']
                answers = row['answer_merge']
                for answer in answers:
                    file.write(f"  {intent_name}:\n  - {answer}\n")


    def __write_config(self, df, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            # Ваша логика для записи конфигурационных данных (если есть)

# Пример использования
generator = YourRasaConfigGenerator()
generator.__write_headers()
