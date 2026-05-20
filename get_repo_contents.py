import urllib.request
import json

def get_contents_for_path(path=""):
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
    print("Exploring uoftcprg/phh-dataset...")
    print("=" * 60)
    
    # Check data folder
    print("\n--- Files in 'data/' folder ---")
    data_contents = get_contents_for_path("data")
    for item in data_contents:
        print(f"- Name: {item['name']} ({item['type']})")
        if item['type'] == 'dir':
            # Sub-contents of directories in data
            sub_contents = get_contents_for_path(f"data/{item['name']}")
            print(f"  [Folder: data/{item['name']} contains {len(sub_contents)} items]")
            for sub_item in sub_contents[:5]:
                print(f"    * {sub_item['name']} ({sub_item['type']})")
            if len(sub_contents) > 5:
                print(f"    * ... and {len(sub_contents) - 5} more files")
        else:
            print(f"  Download URL: {item['download_url']}")
            
    # Check scripts folder
    print("\n--- Files in 'scripts/' folder ---")
    scripts_contents = get_contents_for_path("scripts")
    for item in scripts_contents:
        print(f"- Name: {item['name']} ({item['type']})")
        if item['type'] == 'file':
            print(f"  Download URL: {item['download_url']}")

if __name__ == '__main__':
    main()
