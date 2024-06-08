import json

def save_data2(data):
    with open('data2.json', 'w') as f:
        json.dump(data, f)

# 예시 데이터
data_to_save2 = {
    'B_API': 'asdf',
    'B_SEC': 'asdf',
    'S3_API': 'asdf',
    'S3_SEC': 'asdf',
    'STOKEN': 'asdf'
    }

if __name__ == "__main__":
    save_data2(data_to_save2)
