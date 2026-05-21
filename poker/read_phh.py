import urllib.request

def main():
    url = "https://raw.githubusercontent.com/uoftcprg/phh-dataset/main/data/dwan-ivey-2009.phh"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    try:
        print("Downloading dwan-ivey-2009.phh...")
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                content = response.read().decode('utf-8')
                lines = content.split('\n')
                print(f"Total lines in file: {len(lines)}")
                print("\nFirst 50 lines of dwan-ivey-2009.phh:")
                print("=" * 60)
                for i, line in enumerate(lines[:50]):
                    print(f"{i+1:2d}: {line}")
            else:
                print(f"Error: {response.getcode()}")
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == '__main__':
    main()
