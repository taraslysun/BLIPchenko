import requests
from bs4 import BeautifulSoup
import time

# Constants
URL = "https://virshi.com.ua/ukrainski-poeti-klasiki/"
HEADERS = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
MAX_POEMS = 10000

def create_session():
    """Create and return a persistent session for better performance"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def extract_authors_links(session):
    """Extract author links using the provided session"""
    try:
        response = session.get(URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')  # html.parser is faster than html5lib
        authors_table = soup.find('div', class_='taxonomy-description').find('table')
        return [author['href'] for author in authors_table.find_all('a')]
    except Exception as e:
        print(f"Error extracting author links: {e}")
        return []

def extract_poems_links(session, url):
    """Extract poem links using the provided session"""
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        poems_container = soup.find('div', class_='posts-container')
        if not poems_container:
            return []
        return [poem['href'] for poem in poems_container.find_all('a')]
    except Exception as e:
        print(f"Error extracting poem links from {url}: {e}")
        return []

def extract_poem_text(session, url):
    """Extract poem text using the provided session"""
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        poem_div = soup.find('div', class_='poem-text')
        if not poem_div:
            return ""
        
        # More efficient way to join paragraphs
        # paragraphs = [p.text for p in poem_div.find_all('p')]
        poem_text = []
        for child in poem_div.children:
            if child.name == 'h2':
                break
            if child.name == 'p':
                poem_text.append(child.get_text())
        return "<" + "\n".join(poem_text) + ">"
    except Exception as e:
        print(f"Error extracting poem text from {url}: {e}")
        return ""

def main():
    start_time = time.time()
    print("Starting poem collection...")
    
    # Create a persistent session
    session = create_session()
    
    # Get all author links
    authors_links = extract_authors_links(session)
    print(f"Found {len(authors_links)} authors")
    
    poem_count = 0
    with open("poems.txt", "w", encoding="utf-8") as file:
        # Process each author
        for author_link in authors_links:
            if poem_count >= MAX_POEMS:
                break
                
            # Get poem links for this author
            poems_links = extract_poems_links(session, author_link)
            print(f"Found {len(poems_links)} poems for author at {author_link}")
            
            # Process each poem
            for poem_link in poems_links:
                if poem_count >= MAX_POEMS:
                    break
                    
                # Get poem text
                poem_text = extract_poem_text(session, poem_link)
                if poem_text:
                    file.write(poem_text)
                    poem_count += 1
                    print(f"Collected poem {poem_count}/{MAX_POEMS}")
    
    elapsed_time = time.time() - start_time
    print(f"Scraping completed. Collected {poem_count} poems in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()