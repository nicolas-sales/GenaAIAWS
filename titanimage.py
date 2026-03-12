import boto3 # Permet au programme de parler avec les services AWS, ici Bedrock
import json # Transformer un dictionnaire Python en JSON et lire la réponse JSON renvoyée par Bedrock
import base64
import os

# Les modèles d’image renvoient l’image encodée en base64.
# Base64 = texte qui représente un fichier binaire.
# On doit décoder ce texte pour reconstruire l’image.

prompt_data="""
Ultra realistic cinematic photograph of a tropical beach during rainy season,
dramatic clouds, wet sand reflections, soft light, 4k photography,
high detail, professional photography
"""

bedrock=boto3.client(service_name="bedrock-runtime") # Création d'un client AWS Bedrock Runtime

payload= {
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt_data
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 1024,
        "width": 1024,
        "cfgScale": 8, # Contrôle à quel point l’image doit respecter le prompt, plus la valeur est grande, plus l’image suit le prompt
        "seed": 0
    }
}

body=json.dumps(payload) # Convertit le dictionnaire Python payload en chaîne JSON
model_id="amazon.titan-image-generator-v2:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json", # Recevoir la réponse au format JSON
    contentType="application/json" # Envoi du JSON
)

# Lecture de la réponse
response_body=json.loads(response.get("body").read())
print(response_body)

# Récupération de l'image en base64
image_base64 = response_body.get("images")[0]

# Décodage base64 → image
image_bytes = base64.b64decode(image_base64)

# Sauvegarde de l'image
output_dir="output"
os.makedirs(output_dir,exist_ok=True) # Crée un dossier s'il n'existe pas
file_name=f"{output_dir}/generated-img.png" # Construction du chemin du fichier
with open(file_name,"wb")as f: # Ouverture du fichier
    f.write(image_bytes) # Ecriture de l'image dans le fichier