import urllib.request
import urllib.parse
import json

def search_github():
    print("Searching GitHub Code Search API for VPIP and PFR in CSV files...")
    
    # Query for 'vpip' and 'pfr' and 'extension:csv'
    q = 'vpip pfr extension:csv'
    url = f'https://api.github.com/search/code?q={urllib.parse.quote(q)}'
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status_code == 200 or response.getcode() == 200:
                html = response.read().decode('utf-8')
                data = json.loads(html)
                items = data.get('items', [])
                print(f"Found {len(items)} matching CSV files on GitHub:")
                for item in items[:8]:
                    print(f"\n- Repo: {item['repository']['full_name']}")
                    print(f"  Path: {item['path']}")
                    # Format raw URL
                    # From: https://github.com/user/repo/blob/master/path/to/file.csv
                    # To: https://raw.githubusercontent.com/user/repo/master/path/to/file.csv
                    html_url = item['html_url']
                    raw_url = html_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                    print(f"  Raw URL: {raw_url}")
            else:
                print(f"GitHub API Error: {response.getcode()}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode('utf-8'))
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    search_github()
