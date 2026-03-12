import boto3 # Permet au programme de parler avec les services AWS, ici Bedrock
import json # Transformer un dictionnaire Python en JSON et lire la réponse JSON renvoyée par Bedrock

prompt_data="""
Act as a Shakespeare and write a poem on Generative AI
"""

bedrock=boto3.client(service_name="bedrock-runtime") # Création d'un client AWS Bedrock Runtime

# Requête à envoyer au modèle
payload={
    "prompt":prompt_data,
    "max_tokens_to_sample":512,
    "temperature":0.8,
    "top_p":0.8 # Limite le choix des mots aux plus probables
}

body=json.dumps(payload) # Convertit le dictionnaire Python payload en chaîne JSON
model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json", # Recevoir la réponse au format JSON
    contentType="application/json" # Envoi du JSON
)

response_body=json.loads(response.get("body").read()) # Récupère et Transforme le JSON reçu en dictionnaire Python
response_text=response_body.get["generation"][0].get("data").get("text")
print(response_text)