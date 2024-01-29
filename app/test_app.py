import requests
input_text = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский, после чего поселковый совет стал называться Советским.'
response = requests.get("http://localhost:8000/simplify/", params={"text": input_text})
assert response.json()["text"] == input_text
print(response.json()["simplified_text"])