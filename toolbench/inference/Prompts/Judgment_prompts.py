JUDGMENT_SYSTEM_PROMPT = """Given a user query which should be resolved using an external API. Then, given the documentation of the external API, the input parameters for calling the API and the returned results given by the API. Please judge whether the returned result contains any valuable information. You should first explain your reason and then answer "Yes" or "No".

Specifically, you should answer "No" if the returned result does not provide any valuable information or only contains error messages. You should answer "Yes" if the returned result contains some valuable information other than error messages.

Now, judge whether the returned result is helpful for resolving the user query in the case below. Please give your reason in `reason` and your judgment in `judgment` of JSON to `give_judgment`."""

JUDGMENT_USER_PROMPT = """
User query: {input_description}
API documentation: {api_documentation}
Input parameters: {function_input}
Returned result: {observation}
"""
