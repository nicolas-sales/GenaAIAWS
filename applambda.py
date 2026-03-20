import boto3
import botocore.config
import json

def blog_generate_using_bedrock(blogtopic:str)-> str:
    prompt=f"""                                      
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{blogtopic}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    body={
        "prompt":prompt,
        "max_gen_len":512,
        "temperature":0.5,
        "top_p":0.9 # Limite le choix des mots aux plus probables
    }

    try:
        bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1",config=botocore.config.Config(read_timeout=300,retries={"max_attempts":3}))

        response=bedrock.invoke_model(
        body=json.dumps(body),
        modelId="us.meta.llama3-1-8b-instruct-v1:0")

        response_content=response.get("body").read()
        response_data=json.loads(response_content)
        print(response_data)
        blog_details=response_data["generation"]
        return blog_details

    except Exception as e:
        print(f"Error generating the blog:{e}")
        return ""

def lambda_handler(event, context): # event: données d'entrées, context: infos techniques sur l'exécution
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
