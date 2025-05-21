from openai import OpenAI


API_SECRET_KEY = "Use your own API_SECRET_KEY"
BASE_URL = "Use your own BASE_URL"


# chat
def get_weights_by_gpt(query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in federated learning and federated distillation. "
                           "Analyze and compute the optimal weight allocation for all clients based on the following data in a federated distillation scenario.   Return only the numerical values separated by commas, without any explanation or calculation process. "
            },
            {"role": "user",
             "content": query +
                        "\nBased on these factors, please compute and normalize the weight for each client to maximize the global model performance.\n" +
                        "Assign weights to each client and return the values separated by commas. Do not include any additional text or calculation details:"}
        ]
    )
    print(f"gpt ----- resp:{resp}")
    return resp.choices[0].message.content


if __name__ == '__main__':
    query = "This is the first time I've accessed you via api"
    get_weights_by_gpt(query)