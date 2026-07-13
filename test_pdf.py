import requests

url = "http://127.0.0.1:8000/api/evaluations/6/pdf"
print("Requesting PDF report from API...")
try:
    response = requests.get(url, timeout=30)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Success! Saving PDF file...")
        with open("test_report.pdf", "wb") as f:
            f.write(response.content)
        print("PDF saved as test_report.pdf")
    else:
        print("Failed! Response content:")
        print(response.text)
except Exception as e:
    print("Error during request:", e)
