import json
from os import path

class PromptGenerator:
    def __init__(self, model_name: str, config_location: str = ''):
        """
        Initialize the PromptGenerator with the specified model_name.
        """
        self.model_name = model_name
        
        if config_location:
            config_file_path = config_location
        else:
            config_file_path = path.join(path.dirname(path.realpath(__file__)), "", "prompt_templates.json")
        with open(config_file_path, 'r') as f:
            self.model_templates = json.load(f)

        self.conversation = []
        self.system_text = ""

    def __len__(self):
        """
        length of the conversation items.

        Args:
            template_name (str): The name of the predefined template to load.

        Return:
            the length of the conversation.
        """
        return len(self.conversation)

    def load_template(self, template: str):
        """
        Load a predefined template for the conversation based on the provided template_name.

        Args:
            template_name (str): The name of the predefined template to load.

        Raises:
            ValueError: If the specified template_name is not found.
        """
        template = template.lower()
        if template in self.model_templates:
            self.model_prompt_helper = self.model_templates[template]['response']
            self.user_prompt_helper = self.model_templates[template]['user']
            self.system_prompt_helper = self.model_templates[template]['system']
            self.input_prompt_helper = self.model_templates[template]['input']
            self.template_name = template
            if self.template_name == "llama-2-chat":
                self.is_llama_2 = True
            else:
                self.is_llama_2 = False
        else:
            raise ValueError("Template not found.")

    def custom_template(
        self, system_template: str, response_template: str, user_template: str, input_template: str = ''
    ):
        """
        Create a custom template for the conversation.

        Args:
            system_template (str): The template for the system prompt.\n
            user_template (str): The template for the user prompt.\n
            response_template (str): The template for the model prompt.\n
            input_template (str): The template for the user prompt.\n
            example:\n
            vicuna: {
                    "system": "{system}\\n",
                    "user": "USER: {prompt}\\nASSISTANT:",
                    "response": " {response}</s>",
                    "input": ""
                },

        """
        self.model_prompt_helper = response_template
        self.user_prompt_helper = user_template
        self.system_prompt_helper = system_template
        self.input_prompt_helper = input_template

    def set_system_prompt(self, system_text: str):
        """
        Set the system prompt text in the conversation.

        Args:
            system_text (str): The text of the system prompt.
        """
        self.conversation = []
        self.system_text = system_text
        if self.template_name == "openai":
            self.conversation.append(
                {"role": f"{self.system_prompt_helper}", "content": self.system_text}
            )
        else:
            self.conversation.append(
                self.system_prompt_helper.format(system=self.system_text.strip())
            )

    def add_to_conversation(self, role: str, text: str, preprompt: str = "", input: str = ""):
        """
        Add a prompt to the conversation based on the role.

        Args:
            role (str): The role of the prompt (system, user, or model).
            text (str): The text of the prompt.
            preprompt (str): The text to be added before of the prompt(whitespace is not added).
            input (str): Used for the models that have input template (alpaca), Only use it with the user role.

        Raises:
            ValueError: If the role is not one of 'system', 'user', or 'model'.
        """
        if role != 'user' and input:
            raise ValueError("Input parameter can only be used with user role")
        
        if role.lower() == "system":
            self.set_system_prompt(preprompt.strip() + text.strip())
            
        elif role.lower() == "user":
            if len(self.conversation) == 1 and self.is_llama_2:
                self.conversation.append(self.user_prompt_helper.format(prompt=preprompt.strip() + text.strip()).replace(' [INST]',''))
            elif self.template_name == "openai":
                self.conversation.append(
                    {
                        "role": f"{self.user_prompt_helper}",
                        "content": preprompt.strip() + text.strip(),
                    }
                )
            elif input and self.template_name == "alpaca":
                self.conversation.append(self.input_prompt_helper.format(prompt=preprompt.strip() + text.strip()))
            else:
                self.conversation.append(self.user_prompt_helper.format(prompt=preprompt.strip() + text.strip(), input = input.strip()))

        elif role.lower() == "model":
            if self.template_name == "openai":
                self.conversation.append(
                    {
                        "role": f"{self.model_prompt_helper}",
                        "content": preprompt.strip() + text.strip(),
                    }
                )
            else:
                self.conversation.append(self.model_prompt_helper.format(response=preprompt.strip() + text.strip()))

    def clear_conversation(self):
        """
        Clear the conversation history.
        """
        self.conversation = self.conversation[:1]

    def reduce_length(self, conv: int = 4):
        """
        reduce the conversation history, it reduce the conversation size to the system prompt and the last n conversations.
        Args:
            conv (int): The number of conversation(user prompt and the model answer), including the incomplete one.

        """
        couple = conv * 2 - 1
        if len(self.conversation) < couple:
            couple = len(self.conversation) - 1
        self.conversation = self.conversation[:1] + self.conversation[-couple:]

    def generate_one_shot_prompt(
        self, user_prompt: str, preprompt: str = "", system_prompt: str = "", input: str = ""
    ):
        """
        Generate a one-shot conversation prompt.

        Args:
            user_prompt (str): The text of the user prompt.
            preprompt (str): The text added before the user prompt (No Whitespace).
            system_prompt (str): The text of the system prompt.
            input (str): Used for the models that have input template (alpaca).

        Returns:
            str: The formatted one-shot conversation prompt.
        """
        
        if system_prompt:
            self.system_text = system_prompt.strip()
        # else:
        #     if not self.system_text:
        #         raise ValueError("System prompt not set, please set it by passing it as a parameter or by using .set_system_prompt method.")
        if self.model_name == 'alpaca' and input:
            prompt = (
            self.system_prompt_helper.format(system=self.system_text.strip())
            + self.input_prompt_helper.format(prompt=preprompt.strip() + user_prompt.strip(), input=input.strip())
        )
        else:
            prompt = (
                self.system_prompt_helper.format(system=self.system_text.strip())
                + self.user_prompt_helper.format(prompt=preprompt.strip() + user_prompt.strip())
            )
        return prompt.strip()

    def generate_prompt(self):
        """
        Generate the conversation prompt based on the conversation history.

        Returns:
            str: The formatted conversation prompt.
        """
        if self.template_name == "openai":
            return self.conversation
        else:
            return "".join(self.conversation).strip()


# if __name__ == "__main__":
#     model_name = "gpt2"  # You can change this to other models like "gpt2-medium", "gpt2-large", etc.
#     prompt_generator = PromptGenerator(model_name)

#     # Load predefined template
#     template_name = "orca-hashes"
#     prompt_generator.load_template(template_name)

#     # Test predefined template
#     print("Testing Predefined Template:")
#     print("Loaded Template:", template_name)

#     # Add conversation history
#     prompt_generator.set_system_prompt("I'm system")
#     prompt_generator.add_to_conversation("User", "Tell me a joke.1")
#     prompt_generator.add_to_conversation("model", "NO!!.2")
#     prompt_generator.add_to_conversation("User", "fuckk youu3.")
#     prompt_generator.add_to_conversation("model", "fucker4")
#     prompt_generator.add_to_conversation("User", "fuckk youu22222.5")
#     prompt_generator.add_to_conversation("User", "Tell me a joke.6")
#     prompt_generator.add_to_conversation("model", "NO!!.7")
#     prompt_generator.add_to_conversation("User", "fuckk youu.8")
#     prompt_generator.add_to_conversation("model", "fucker9")
#     prompt_generator.add_to_conversation("User", "fuckk youu22222.9")
#     prompt_generator.add_to_conversation("model", "NO!!.11")
#     prompt_generator.add_to_conversation("User", "fuckk youu.12")
#     prompt_generator.add_to_conversation("model", "fucker13")
#     prompt_generator.add_to_conversation("User", "fuckk youu22222.14")
#     prompt_generator.reduce_length(3)
#     prompt = prompt_generator.generate_prompt()
#     print(f"{template_name} Conversation:\n\n{prompt}\n")

#     # Clear conversation history
#     prompt_generator.clear_conversation()
#     prompt = prompt_generator.generate_prompt()
#     print(f"{template_name} Conversation:\n{prompt}\n")
#     prompt_generator.add_to_conversation("User", "Tell me a joke.1")
#     prompt = prompt_generator.generate_prompt()
#     print(f"{template_name} Conversation:\n{prompt}\n")
#     # Test custom template
#     # custom_template = "CUSTOM TEMPLATE\n{conversation}\n"
#     # prompt_generator.custom_template(custom_template)

#     # print("Testing Custom Template:")
#     # print("Custom Template:")
#     # print(custom_template)

#     # # Add conversation history
#     # prompt_generator.add_to_conversation("System", "I'm a knowledgeable assistant.")
#     # prompt_generator.add_to_conversation("User", "What's the capital of France?")
#     # prompt_generator.add_to_conversation("Model", "The capital of France is Paris.")
#     # prompt = prompt_generator.generate_prompt()
#     # print("Custom Conversation:\n", prompt)
