import json

def save_data(data):
    with open('data.json', 'w') as f:
        json.dump(data, f)

# 예시 데이터
data_to_save = {
    'B_API': 'B_API',
    'B_SEC': 'B_SEC',
    'S3_API': 'S3_API',
    'S3_SEC': 'S3_SEC',
    'STOKEN': 'S_TOKEN'
    }

if __name__ == "__main__":
    save_data(data_to_save)
