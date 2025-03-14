import os
import re

# Set the target directory (use "." for the current directory)
target_directory = "."

def sanitize_display_name(filename):
    """Modifies display name:
    1. If the last space is followed by 20+ alphanumeric characters, remove it and everything after.
    2. Keeps the original file extension.
    """
    # Find the last space in the filename
    last_space_index = filename.rfind(" ")
    
    if last_space_index != -1:
        # Check if the part after the last space has 20+ alphanumeric characters
        after_space = filename[last_space_index + 1:]
        if re.match(r'^[a-zA-Z0-9]{20,}$', after_space):
            filename = filename[:last_space_index]  # Remove last space and everything after

    return filename

def generate_markdown(directory, base_path=""):
    markdown_content = []
    
    for root, dirs, files in os.walk(directory):
        # Ignore hidden files and directories (like .git)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files = [f for f in files if f.endswith('.md') or f.endswith('.pdf')]

        # Skip empty folders (without .md or .pdf files)
        if not files:
            continue

        # Compute indentation level
        level = root.replace(directory, "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        
        # Add folder to markdown
        markdown_content.append(f"{indent}- **{folder_name}/**")
        
        # Add files and generate relative links
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), base_path)
            link_path = file_path.replace(" ", "%20")  # Convert spaces to %20 in link
            display_name = sanitize_display_name(file)
            markdown_content.append(f"{indent}  - [{display_name}]({link_path})")
    
    return "\n".join(markdown_content)

# Generate README.md
output_file = "README.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Project Directory Structure\n\n")
    f.write(generate_markdown(target_directory))

print(f"✅ Markdown directory generated! Check {output_file}")
