import os
import re

# Set the target directory (use "." for the current directory)
target_directory = "."

def sanitize_display_name(name):
    """Sanitizes display name for both files and folders:
    1. Replaces various space-like characters (e.g., em dash) with a standard space.
    2. If the last space is followed by 20+ alphanumeric characters, remove the last space and everything after.
    3. Keeps the original file extension intact (for files).
    """
    # Normalize spaces (replace non-standard spaces with regular space)
    name = name.replace("—", " ").replace("–", " ")

    # Split name into base and extension
    base_name, ext = os.path.splitext(name)

    # Find the last space in the base name
    last_space_index = base_name.rfind(" ")

    if last_space_index != -1:
        # Check if the part after the last space is 20+ alphanumeric characters
        after_space = base_name[last_space_index + 1:]
        if re.match(r'^[a-zA-Z0-9]{20,}$', after_space):
            base_name = base_name[:last_space_index]  # Remove last space and everything after

    return base_name + ext  # Reattach the extension

def generate_markdown(directory, base_path=""):
    markdown_content = []
    
    for root, dirs, files in os.walk(directory):
        # Ignore hidden files and directories (like .git)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files = [f for f in files if f.endswith(('.md', '.pdf', '.html'))]  # Now includes .html files

        # Skip empty folders (without .md, .pdf, or .html files)
        if not files and not dirs:
            continue

        # Compute indentation level
        level = root.replace(directory, "").count(os.sep)
        indent = "  " * level

        # Get folder name and shorten if needed
        folder_name = os.path.basename(root)
        display_folder_name = sanitize_display_name(folder_name)

        # Add folder to markdown
        markdown_content.append(f"{indent}- **{display_folder_name}/**")

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
