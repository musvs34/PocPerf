# test_salesforce_connection.py
import requests
import json

DOMAIN          = "carac--int.sandbox.my.salesforce.com"
CONSUMER_KEY    = "<YOUR_CONSUMER_KEY>"
CONSUMER_SECRET = "<YOUR_CONSUMER_SECRET>"

TOKEN_URL = f"https://{DOMAIN}/services/oauth2/token"

def get_access_token():
    payload = {
        "grant_type":    "client_credentials",
        "client_id":     CONSUMER_KEY,
        "client_secret": CONSUMER_SECRET,
    }
    resp = requests.post(TOKEN_URL, data=payload)
    resp.raise_for_status()
    data = resp.json()
    print(" Token obtenu")
    print(f"   Instance URL : {data['instance_url']}")
    print(f"   Token type   : {data['token_type']}")
    return data

def test_api_call(instance_url, token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{instance_url}/services/data/v59.0/sobjects/"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    sobjects = resp.json().get("sobjects", [])
    print(f"\n✅ API REST OK — {len(sobjects)} objets disponibles")
    print("   Exemples :", [o["name"] for o in sobjects[:5]])

if __name__ == "__main__":
    try:
        token_data = get_access_token()
        test_api_call(token_data["instance_url"], token_data["access_token"])
    except requests.exceptions.HTTPError as e:
        print(f" Erreur HTTP : {e.response.status_code}")
        print(e.response.text)
    except Exception as e:
        print(f" Erreur : {e}")
