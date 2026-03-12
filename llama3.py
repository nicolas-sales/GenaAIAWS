import boto3 # Permet au programme de parler avec les services AWS, ici Bedrock
import json # Transformer un dictionnaire Python en JSON et lire la réponse JSON renvoyée par Bedrock

prompt_data="""
Act as a Shakespeare and write a poem on Generative AI
"""

# Reformatage du prompt pour qu’il respecte le format attendu par Llama 3.1. Le f permet d'insérer la variable {prompt_data}
formatted_prompt = f"""                                      
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt_data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bedrock=boto3.client(service_name="bedrock-runtime") # Création d'un client AWS Bedrock Runtime

# Requête à envoyer au modèle
payload={
    "prompt":formatted_prompt,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9 # Limite le choix des mots aux plus probables
}

body=json.dumps(payload) # Convertit le dictionnaire Python payload en chaîne JSON
model_id="us.meta.llama3-1-8b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json", # Recevoir la réponse au format JSON
    contentType="application/json" # Envoi du JSON
)

response_body=json.loads(response.get("body").read()) # Récupère et Transforme le JSON reçu en dictionnaire Python
response_text=response_body["generation"]
print(response_text)