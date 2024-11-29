import requests
from bs4 import BeautifulSoup
import re

def fetch_metadata(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No Title Found"

        # Attempt to extract author and year from meta tags
        author = "Unknown Author"
        year = "Unknown Year"

        # Look for common meta tags for authors
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and 'content' in author_meta.attrs:
            author = author_meta['content']
        
        # Look for common meta tags for publication date
        date_meta = soup.find('meta', attrs={'name': 'date'}) or soup.find('meta', attrs={'property': 'article:published_time'})
        if date_meta and 'content' in date_meta.attrs:
            year = date_meta['content'][:4]  # Get the year from the date string

        # Check for "Written by" pattern in page content
        if author == "Unknown Author":  # Only search if we haven't found an author yet
            text = soup.get_text()
            match = re.search(r'Written by\s*([^,]+)', text)
            if match:
                author = match.group(1).strip()

        return {
            'title': title,
            'author': author,
            'year': year,
            'source': url
        }
        
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def create_ris(metadata, output_file):
    if metadata:
        with open(output_file, 'w') as f:
            f.write("TY  - GEN\n")  # Type of reference
            f.write(f"TI  - {metadata['title']}\n")
            f.write(f"AU  - {metadata['author']}\n")
            f.write(f"PY  - {metadata['year']}\n")
            f.write(f"UR  - {metadata['source']}\n")
            f.write("ER  - \n")  # End of reference
        print(f"RIS file created: {output_file}")
    else:
        print("No metadata to write.")

def main():
    url = input("Enter the URL: ")
    metadata = fetch_metadata(url)
    create_ris(metadata, "output.ris")

if __name__ == "__main__":
    main()
