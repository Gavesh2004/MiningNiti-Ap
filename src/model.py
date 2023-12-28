import os
from typing import Any
#import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class MyChatBot:
    def __init__(self, api_key, temperature):
        os.environ['GOOGLE_API_KEY'] = api_key
        #genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
        self.chain = None

    def set_prompt(self, template, input_variables):
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def run(self, input):
        if self.chain is not None:
            return self.chain.predict(input=input)
        else:
            raise Exception("Prompt not set. Please set the prompt using set_prompt method.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
