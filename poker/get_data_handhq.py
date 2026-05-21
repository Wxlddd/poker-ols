import urllib.request
import json

def get_contents_for_path(path):
    url = f"https://api.github.com/repos/uoftcprg/phh-dataset/contents/{path}"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to fetch {path}: {e}")
    return []

def main():
    path = "data/handhq/ABS-2009-07-01_2009-07-23_100NLH_OBFU"
    print(f"Inspecting folder: {path}")
    print("=" * 60)
    contents = get_contents_for_path(path)
    print(f"Folder contains {len(contents)} items:")
    for item in contents[:10]:
        print(f"- Name: {item['name']} ({item['type']})")
        if item['type'] == 'file':
            print(f"  Size: {item['size']} bytes")
            print(f"  Download URL: {item['download_url']}")
    if len(contents) > 10:
        print(f"... and {len(contents) - 10} more files")

if __name__ == '__main__':
    main()
