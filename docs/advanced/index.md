---
title: Advanced
description: Advanced usage patterns
icon: material/file-code
order: 6
hide:
  - toc
---

````python exec="true"
import pathlib
import re
from typing import Dict, Any
from jinjarope.iconfilters import get_icon_svg
def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Parse YAML frontmatter from markdown content."""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}
    
    frontmatter = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip()] = value.strip().strip('"\'')
    return frontmatter

# Collect page data
pages = []
for page_path in pathlib.Path("docs/advanced").iterdir():
    if page_path.suffix == '.md' and page_path.name != 'index.md':
        content = page_path.read_text()
        frontmatter = parse_frontmatter(content)
        
        # Get title, description, and icon from frontmatter
        title = frontmatter.get('title', page_path.stem.replace('-', ' ').title())
        description = frontmatter.get('description', '')
        icon = frontmatter.get('icon', '')
        order = int(frontmatter.get('order', 999))  # Default high order for unordered items
        
        # Create relative link
        link = page_path.name
        
        pages.append({
            'title': title,
            'description': description, 
            'icon': icon,
            'link': link,
            'order': order
        })

# Sort by order, then by title
pages.sort(key=lambda x: (x['order'], x['title']))

# Generate table
print("| Page | Description |")
print("|------|-------------|")
for page in pages:
    icon_part = f"{get_icon_svg(page['icon'])} " if page['icon'] else ""
    print(f"| {icon_part}[{page['title']}]({page['link']}) | {page['description']} |")

````
