import boto3
import botocore.config
import json
import datatime

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
    
    
def save_blog_details_s3(s3_key,s3_bucket,generate_blog):
    s3=boto3.client("s3")

    try:
        s3.put_object(Bucket=s3_bucket,key=s3_key,Body=generate_blog)
        print("Code saved to s3")

    except Exception as e:
        print("Error when saving the code to s3")


def lambda_handler(event, context): # event: données d'entrées, context: infos techniques sur l'exécution
    # TODO implement
    event=json.loads(event['body'])
    blogtopic=event['blog_topic']

    generate_blog=blog_generate_using_bedrock(blogtopic=blogtopic)

    if generate_blog:
        current_time=datatime.now().strftime("%H%M%S")
        s3_key=f"blog-outputs/{current_time}.txt"
        s3_bucket="aws_bedrock"
        save_blog_details_s3(s3_key,s3_bucket,generate_blog)

    else:
        print('No blog was generated')

    return {
        "statuscode":200,
        "body":json.dumps('Blog Generation is completed')
    }

    
