# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# from typing import Any, Text, Dict, List
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import SlotSet

# class ActionSetUserName(Action):
#     def name(self) -> Text:
#         return "action_set_user_name"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         # Получаем имя пользователя из сущности "user_name"
#         user_name = next(tracker.get_latest_entity_values("user_name"), None)
        
#         if not user_name:
#             # Если сущность не найдена, используем значение "User" по умолчанию
#             user_name = "User"

#         # Устанавливаем значение слота
#         return [SlotSet("user_name", user_name)]

# class ActionGreetWithName(Action):
#     def name(self) -> Text:
#         return "action_greet_with_name"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,   
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         # Получаем имя пользователя из слота
#         user_name = tracker.get_slot("user_name")

#         if user_name:
#             # Если имя есть, обращаемся по имени
#             response = f"Привет, {user_name}!"
#         else:
#             # Если имя неизвестно, используем общий приветственный ответ
#             response = "Привет! Как я могу помочь вам?"

#         dispatcher.utter_message(response)
#         return []


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionGreetWithName(Action):
    def name(self) -> Text:
        return "action_greet_with_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Извлекаем имя из слота user_name
        user_name = tracker.get_slot("user_name")

        # Отправляем приветственное сообщение с использованием имени пользователя
        if user_name:
            message = f"Привет, {user_name}! Как я могу помочь вам сегодня?"
        else:
            message = "Привет! Как я могу помочь вам сегодня?"

        dispatcher.utter_message(text=message)

        return []