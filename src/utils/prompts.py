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
            "content": f"Context: {context} \nQuestion: {question} \nAnswer: "
        }
    ]
    
    return prompt

def get_GRAG_prompt(language):
    if language == "deu":
        return get_Graph_prompt_deu
    elif language == "fra":
        return get_Graph_prompt_fra
    elif language == "spa":
        return get_Graph_prompt_spa
    elif language == "ita":
        return get_Graph_prompt_ita
    elif language == "por":
        return get_Graph_prompt_por
    else:
        return get_Graph_prompt_default
    
def get_QA_prompt(language):
    if language == "deu":
        return get_QA_prompt_deu
    elif language == "fra":
        return get_QA_prompt_fra
    elif language == "spa":
        return get_QA_prompt_spa
    elif language == "ita":
        return get_QA_prompt_ita
    elif language == "por":
        return get_QA_prompt_por
    else:
        return get_QA_prompt_default
    
###
### GRAPH PROMPTS ###
###

def get_Graph_prompt_deu(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""Sie müssen die Frage des Benutzers beantworten. Sie müssen in Ihrer Antwort keine Begründung oder Erklärung liefern, sondern nur die Antwort.
Der Kontext ist ein optionaler Textabschnitt, der hilfreich sein könnte. Sie können ihn bei Bedarf als zusätzliches Wissen verwenden. Ist er nicht hilfreich, können Sie ihn ignorieren und Ihre eigene Vermutung anstellen.

Hier ist ein Beispiel für die Eingabe:
Kontext: Albert Einstein, Geburtsort Ulm, Land Deutschland
Frage: Wo wurde Albert Einstein geboren?

Hier ist ein Beispiel für die Ausgabe:
Antwort: Albert Einstein wurde in Ulm, Deutschland, geboren."""
        },
        {
            "role": "user",
            "content": f"Kontext: {kg_path} \nFrage: {question} \nAntwort:"
        }
    ]
    return prompt

def get_Graph_prompt_fra(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""Vous devez répondre à la question posée par l'utilisateur. Dans votre réponse, vous n'avez pas besoin de fournir de raisonnement ni d'explication, mais simplement de fournir la réponse.
Le chemin est un passage de texte facultatif qui peut être utile. Vous pouvez donc l'utiliser comme information complémentaire si nécessaire. S'il ne vous est pas utile, vous pouvez l'ignorer et faire votre meilleure estimation. 

Voici un exemple de saisie.
Contexte: Albert Einstein lieu de naissance Ulm pays Allemagne
Question: Où est né Albert Einstein?

Voici un exemple de sortie.
Réponse: Albert Einstein est né à Ulm, en Allemagne."""
        },
        {
            "role": "user",
            "content": f"Contexte: {kg_path} \nQuestion: {question} \nRéponse:"
        }
    ]
    return prompt    
    
def get_Graph_prompt_spa(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""Debe responder a la pregunta del usuario. En su respuesta, no necesita proporcionar ningún razonamiento ni explicación, simplemente proporcione la respuesta.
El contexto es un fragmento de texto opcional que puede ser útil. Puede usarlo como información adicional si lo necesita. Si no es útil, puede ignorarlo y hacer su mejor estimación.

Aquí tiene un ejemplo de entrada:
Contexto: Albert Einstein, lugar de nacimiento, Ulm, país, Alemania
Pregunta: ¿Dónde nació Albert Einstein?

Aquí tiene un ejemplo de salida:
Respuesta: Albert Einstein nació en Ulm, Alemania."""
        },
        {
            "role": "user",
            "content": f"Contexto: {kg_path} \nPregunta: {question} \nRespuesta:"
        }
    ]
    return prompt

def get_Graph_prompt_ita(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""Devi rispondere alla domanda posta dall'utente. Nella tua risposta, non è necessario fornire alcuna motivazione o spiegazione, ma solo la risposta.
Il contesto è un testo facoltativo che può essere utile. Puoi utilizzarlo come informazione aggiuntiva, se necessario. Se non è utile, puoi ignorarlo e fare la tua ipotesi migliore.

Ecco un esempio di input.
Contesto: Albert Einstein, luogo di nascita, Ulm, paese, Germania
Domanda: Dove è nato Albert Einstein?

Ecco un esempio di output.
Risposta: Albert Einstein è nato a Ulm, in Germania."""
        },
        {
            "role": "user",
            "content": f"Contesto: {kg_path} \nDomanda: {question} \nRisposta:"
        }
    ]
    return prompt

def get_Graph_prompt_por(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""Deve responder à pergunta feita pelo utilizador. Na sua resposta, não precisa de fornecer qualquer raciocínio ou explicação, basta fornecer a resposta.
O contexto é um texto opcional que pode ser útil. Pode usá-lo como informação adicional, se necessário. Se não for útil, pode ignorar e dar o seu melhor palpite.

Aqui está um exemplo de entrada.
Contexto: Albert Einstein, naturalidade, Ulm, país, Alemanha
Pergunta: Onde nasceu Albert Einstein?

Aqui está um exemplo de saída.
Resposta: Albert Einstein nasceu em Ulm, na Alemanha."""
        },
        {
            "role": "user",
            "content": f"Contexto: {kg_path} \nPergunta: {question} \nResposta: "
        }
    ]
    return prompt



def get_Graph_prompt_default(kg_path, question):
    prompt = [
        {
            "role": "system",
            "content": f"""You need to answer the question given by the user. In your answer you do not need to provide any reasoning or explanation, only provide the answer.
The Path is an optional text passage that could be useful, so you can use it as additional knowledge if necessary, if it is not helpful, you can ignore it and make your best guess.

Here is example input.
Path: Albert Einstein place of birth Ulm country Germany
Question: Where was Albert Einstein born?

Here is example output.
Answer: Albert Einstein was born in Ulm, Germany."""
        },
        {
            "role": "user",
            "content": f"Path: {kg_path} \nQuestion: {question} \nAnswer: "
        }
    ]
    return prompt


### 
### QA PROMPTS ###
###

def get_QA_prompt_deu(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""Sie müssen die Frage des Benutzers beantworten. Beantworten Sie Ihre Frage mit Ihrem internen Wissen und so präzise und prägnant wie möglich.

Hier ist ein Beispiel für eine Eingabe:
Frage: Wo wurde Albert Einstein geboren?

Hier ist ein Beispiel für eine Ausgabe:
Antwort: Albert Einstein wurde in Ulm geboren."""
        },
        {
            "role": "user",
            "content": f"Frage: {question} \nAntwort: "
        }
    ]
    return prompt

def get_QA_prompt_fra(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""Vous devez répondre à la question posée par l'utilisateur. Répondez en utilisant vos connaissances internes, avec autant de précision et de concision que possible.

Voici un exemple de saisie.
Question: Où est né Albert Einstein?

Voici un exemple de sortie.
Réponse: Albert Einstein est né à Ulm, en Allemagne."""
        },
        {
            "role": "user",
            "content": f"Question: {question} \nRéponse: "
        }
    ]
    return prompt

def get_QA_prompt_spa(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""Debes responder la pregunta del usuario. Responde usando tu intuición y de la forma más precisa y concisa posible.

Aquí tienes un ejemplo de entrada:
Pregunta: ¿Dónde nació Albert Einstein?

Aquí tienes un ejemplo de salida:
Respuesta: Albert Einstein nació en Ulm, Alemania."""
        },
        {
            "role": "user",
            "content": f"Pregunta: {question} \nRespuesta: "
        }
    ]
    return prompt

def get_QA_prompt_ita(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""Devi rispondere alla domanda posta dall'utente. Rispondi utilizzando le tue conoscenze interne e nel modo più preciso e conciso possibile.

Ecco un esempio di input.
Domanda: Dove è nato Albert Einstein?

Ecco un esempio di output.
Risposta: Albert Einstein è nato a Ulm, in Germania."""
        },
        {
            "role": "user",
            "content": f"Domanda: {question} \nRisposta: "
        }
    ]
    return prompt

def get_QA_prompt_por(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""Precisa de responder à pergunta feita pelo utilizador. Responda utilizando o seu conhecimento interno e da forma mais precisa e concisa possível.

Aqui está um exemplo de entrada.
Pergunta: Onde nasceu Albert Einstein?

Aqui está um exemplo de saída.
Resposta: Albert Einstein nasceu em Ulm, na Alemanha."""
        },
        {
            "role": "user",
            "content": f"Pergunta: {question} \nResposta: "
        }
    ]
    return prompt

def get_QA_prompt_default(context, question):
    """ Context is here only to enable interoperability with other prompts """
    prompt = [
        {
            "role": "system",
            "content": f"""You need to answer the question given by the user. Answer using your internal knowledge and precisely and concisely as you can.
            
Here is example input.
Question: Where was Albert Einstein born?

Here is example output.
Answer: Albert Einstein was born in Ulm, Germany."""
        },
        {
            "role": "user",
            "content": f"Question: {question} \nAnswer: "
        }
    ]
    return prompt