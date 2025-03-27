def get_RAG_prompt(context, question):
    prompt = [
        {
            "role": "system",
            "content": f"""You need to answer the question given by the user. You can use Context as additional knowledge if needed.
            The Context is a text passage that could be useful for answering the question. You do not need to provide any reasoning or explanation, only provide the answer.
            
            Here is example input.
            Context: Albert Einstein is a physicist known for developing the theory of relativity. He was born in Ulm, Germany and studied at the Polytechnic Institute in Zurich. He later died in Princeton, New Jersey, USA.
            Question: Where was Albert Einstein born?
            
            Here is example output.
            Answer: Albert Einstein was born in Ulm, Germany."""
        },
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {question}\nAnswer: "
        }
    ]
    
    return prompt

def get_Graph_prompt(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""You need to answer the question given by the user. You can use Path as additional knowledge if needed.
            The Path describes a knowledge graph path taken from Wikidata. The path entities and relations are seperated by semicolon. You do not need to provide any reasoning or explanation, only provide the answer.
            
            Here is example input.
            Path: Albert Einstein; place of birth; Ulm; country; Germany
            Question: Where was Albert Einstein born?
            
            Here is example output.
            Answer: Albert Einstein was born in Ulm, Germany.
            """
        },
        {
            "role": "user",
            "content": f"Path: {kg_path}\nQuestion: {question}\nAnswer: "
        }
    ]
    return prompt

def get_standard_QA_prompt(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""You need to answer the question given by the user. Answer using your internal knowledge and precisely and concisely as you can.
            
            Here is example input.
            Question: Where was Albert Einstein born?
            
            Here is example output.
            Answer: Albert Einstein was born in Ulm, Germany.
            """
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer: "
        }
    ]
    return prompt