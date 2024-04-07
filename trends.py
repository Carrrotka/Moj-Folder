import http.client

conn = http.client.HTTPSConnection("dataforseo-search-volume1.p.rapidapi.com")

payload = "[\n    {\n        \"keywords\": [\n            \"average page rpm adsense\",\n            \"adsense blank ads how long\",\n            \"leads and prospects\"\n        ],\n        \"language_name\": \"English\",\n        \"location_name\": \"United States\",\n        \"search_partners\": true,\n        \"sort_by\": \"search_volume\"\n    }\n]"

headers = {
    'content-type': "application/json",
    'Content-Type': "application/json",
    'Authorization': "<REQUIRED>",
    'X-RapidAPI-Key': "4d2d836847msh8e6616e60f02ea8p1873a1jsnf03e051a5c2d",
    'X-RapidAPI-Host': "dataforseo-search-volume1.p.rapidapi.com"
}

conn.request("POST", "/v3/keywords_data/google/search_volume/live", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))