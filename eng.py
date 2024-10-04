import os
import fnmatch
import logging
import time
from openai import OpenAI
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
import difflib
import re
import argparse

CREATE_SYSTEM_PROMPT = """You are an advanced Software engineer designed to create files and folders based on user instructions. Your primary objective is to generate the content of the files to be created as code blocks. Each code block should specify whether it's a file or folder, along with its path.

When given a user request, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to create.
2. Generate Creation Instructions: Provide the content for each file to be created within appropriate code blocks. Each code block should begin with a special comment line that specifies whether it's a file or folder, along with its path.
3. You create full functioning, complete,code files, not just snippets. No approximations or placeholders. FULL WORKING CODE.

IMPORTANT: Your response must ONLY contain the code blocks with no additional text before or after. Do not use markdown formatting outside of the code blocks. Use the following format for the special comment line. Do not include any explanations, additional text:

For folders:
```
### FOLDER: path/to/folder
```

For files:
```language
### FILE: path/to/file.extension
File content goes here...
```

Example of the expected format:

```
### FOLDER: new_app
```

```html
### FILE: new_app/index.html
<!DOCTYPE html>
<html>
<head>
    <title>New App</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

```css
### FILE: new_app/styles.css
body {
    font-family: Arial, sans-serif;
}
```

```javascript
### FILE: new_app/script.js
console.log('Hello, World!');
```

Ensure that each file and folder is correctly specified to facilitate seamless creation by the script, and response in Traditional Chinese."""


CODE_REVIEW_PROMPT = """You are an expert code reviewer. Your task is to analyze the provided code files and provide a comprehensive code review. For each file, consider:

1. Code Quality: Assess readability, maintainability, and adherence to best practices
2. Potential Issues: Identify bugs, security vulnerabilities, or performance concerns
3. Suggestions: Provide specific recommendations for improvements

Format your review as follows:
1. Start with a brief overview of all files
2. For each file, provide:
   - A summary of the file's purpose
   - Key findings (both positive and negative)
   - Specific recommendations
3. End with any overall suggestions for the codebase

Your review should be detailed but concise, focusing on the most important aspects of the code."""


EDIT_INSTRUCTION_PROMPT = """You are an advanced Software engineer designed to analyze files and provide edit instructions based on user requests. Your task is to:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the File(s): Review the content of the provided file(s).
3. Generate Edit Instructions: Provide clear, step-by-step instructions on how to modify the file(s) to address the user's request.

Your response should be in the following format:

```
File: [file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...

File: [another_file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...
```

Only provide instructions for files that need changes. Be specific and clear in your instructions."""


APPLY_EDITS_PROMPT = """
Rewrite an entire file or files using edit instructions provided by another AI.

Ensure the entire content is rewritten from top to bottom incorporating the specified changes.

# Steps

1. **Receive Input:** Obtain the file(s) and the edit instructions. The files can be in various formats (e.g., .txt, .docx).
2. **Analyze Content:** Understand the content and structure of the file(s).
3. **Review Instructions:** Carefully examine the edit instructions to comprehend the required changes.
4. **Apply Changes:** Rewrite the entire content of the file(s) from top to bottom, incorporating the specified changes.
5. **Verify Consistency:** Ensure that the rewritten content maintains logical consistency and cohesiveness.
6. **Final Review:** Perform a final check to ensure all instructions were followed and the rewritten content meets the quality standards.
7. Do not include any explanations, additional text, or code block markers (such as ```html or ```).

Provide the output as the FULLY NEW WRITTEN file(s).
NEVER ADD ANY CODE BLOCK MARKER AT THE BEGINNING OF THE FILE OR AT THE END OF THE FILE (such as ```html or ```). 

"""


PLANNING_PROMPT = """You are an AI planning assistant. Your task is to create a detailed plan based on the user's request. Consider all aspects of the task, break it down into steps, and provide a comprehensive strategy for accomplishment. Your plan should be clear, actionable, and thorough."""


last_ai_response = None
conversation_history = []

def is_binary_file(file_path):
    """Check if a file is binary by reading a small portion of it."""
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)  # Read the first 1024 bytes
            if b'\0' in chunk:
                return True  # File is binary if it contains null bytes
            # Use a heuristic to detect binary content
            text_characters = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)))
            non_text = chunk.translate(None, text_characters)
            if len(non_text) / len(chunk) > 0.30:
                return True  # Consider binary if more than 30% non-text characters
    except Exception as e:
        logging.error(f"錯誤讀取檔案 {file_path}: {e}")
        return True  # Assume binary if an error occurs
    return False  # File is likely text


# Load .gitignore patterns if in a git repository
def load_gitignore_patterns(directory):
    gitignore_path = os.path.join(directory, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns

def should_ignore(file_path, patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def add_file_to_context(file_path, added_files, action='to the chat context'):
    """Add a file to the given dictionary, applying exclusion rules."""
    excluded_dirs = {
    '__pycache__',
    '.git',
    'node_modules',
    'venv',
    'env',
    '.vscode',
    '.idea',
    'dist',
    'build',
    '__mocks__',
    'coverage',
    '.pytest_cache',
    '.mypy_cache',
    'logs',
    'temp',
    'tmp',
    'secrets',
    'private',
    'cache',
    'addons'
    }
    # Removed reliance on 'excluded_extensions' and 'supported_extensions'

    # Load .gitignore patterns if in a git repository
    gitignore_patterns = []
    if os.path.exists('.gitignore'):
        gitignore_patterns = load_gitignore_patterns('.')

    if os.path.isfile(file_path):
        # Exclude based on directory
        if any(ex_dir in file_path for ex_dir in excluded_dirs):
            print(f"跳過排除的目錄檔案: {file_path}", "yellow")
            logging.info(f"跳過排除的目錄檔案: {file_path}")
            return
        # Exclude based on gitignore patterns
        if gitignore_patterns and should_ignore(file_path, gitignore_patterns):
            print(f"跳過匹配 .gitignore 模式的檔案: {file_path}", "yellow")
            logging.info(f"跳過匹配 .gitignore 模式的檔案: {file_path}")
            return
        if is_binary_file(file_path):
            print(f"跳過二進制檔案: {file_path}", "yellow")
            logging.info(f"跳過二進制檔案: {file_path}")
            return
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                added_files[file_path] = content
                print(colored(f"Added {file_path} {action}.", "green"))
                print(colored(f"添加 {file_path} {action}.", "green"))
                logging.info(f"添加 {file_path} {action}.")
        except Exception as e:
            print(colored(f"讀取檔案 {file_path} 時發生錯誤: {e}", "red"))
            logging.error(f"讀取檔案 {file_path} 時發生錯誤: {e}")
    else:
        print(colored(f"Error: {file_path} 不是一個檔案.", "red"))
        logging.error(f"{file_path} 不是一個檔案.")



def apply_modifications(new_content, file_path):
    try:
        with open(file_path, 'r') as file:
            old_content = file.read()

        if old_content.strip() == new_content.strip():
            print(colored(f"在 {file_path} 中未檢測到更改", "red"))
            return True

        display_diff(old_content, new_content, file_path)

        confirm = prompt(f"Apply these changes to {file_path}? (yes/no): ", style=Style.from_dict({'prompt': 'orange'})).strip().lower()
        if confirm == 'yes':
            with open(file_path, 'w') as file:
                file.write(new_content)
            print(colored(f"已成功將更改應用於 {file_path}.", "green"))
            logging.info(f"已成功將更改應用於 {file_path}.")
            return True
        else:
            print(colored(f"未將更改應用於 {file_path}.", "red"))
            logging.info(f"用戶選擇不將更改應用於 {file_path}.")
            return False

    except Exception as e:
        print(colored(f"在應用更改到 {file_path} 時發生錯誤: {e}", "red"))
        logging.error(f"在應用更改到 {file_path} 時發生錯誤: {e}")
        return False

def display_diff(old_content, new_content, file_path):
    diff = list(difflib.unified_diff(
old_content.splitlines(keepends=True),
new_content.splitlines(keepends=True),
fromfile=f"a/{file_path}",
tofile=f"b/{file_path}",
lineterm='',
n=5
))
    if not diff:
        print(f"在 {file_path} 中未檢測到更改")
        return
    console = Console()
    table = Table(title=f"Diff for {file_path}")
    table.add_column("Status", style="bold")
    table.add_column("Line")
    table.add_column("Content")
    line_number = 1
    for line in diff:
        status = line[0]
        content = line[2:].rstrip()
        if status == ' ':
            continue  # Skip unchanged lines
        elif status == '-':
            table.add_row("Removed", str(line_number), content, style="red")
        elif status == '+':
            table.add_row("Added", str(line_number), content, style="green")
        line_number += 1
    console.print(table)

def apply_creation_steps(creation_response, added_files, retry_count=0):
    max_retries = 3
    try:
        code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', creation_response)
        if not code_blocks:
            raise ValueError("在 AI 回應中未找到程式碼區塊。")

        print("成功提取程式碼區塊:")
        logging.info("成功從創建回應中提取程式碼區塊。")

        for code in code_blocks:
            # Extract file/folder information from the special comment line
            info_match = re.match(r'### (FILE|FOLDER): (.+)', code.strip())
            
            if info_match:
                item_type, path = info_match.groups()
                
                if item_type == 'FOLDER':
                    # Create the folder
                    os.makedirs(path, exist_ok=True)
                    print(colored(f"創建資料夾: {path}", "green"))
                    logging.info(f"創建資料夾: {path}")
                elif item_type == 'FILE':
                    # Extract file content (everything after the special comment line)
                    file_content = re.sub(r'### FILE: .+\n', '', code, count=1).strip()

                    # Create directories if necessary
                    directory = os.path.dirname(path)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                        print(colored(f"創建資料夾: {directory}", "green"))
                        logging.info(f"創建資料夾: {directory}")

                    # Write content to the file
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(file_content)
                    print(colored(f"創建檔案: {path}", "green"))
                    logging.info(f"創建檔案: {path}")
            else:
                print(colored("錯誤: 無法從程式碼區塊中確定檔案或資料夾資訊。", "red"))
                logging.error("無法從程式碼區塊中確定檔案或資料夾資訊。")
                continue

        return True

    except ValueError as e:
        if retry_count < max_retries:
            print(colored(f"錯誤: {str(e)} 重試... (嘗試 {retry_count + 1})", "red"))
            logging.warning(f"創建解析失敗: {str(e)}. 重試... (嘗試 {retry_count + 1})")
            error_message = f"{str(e)} 請再次提供使用指定格式的創建指令。"
            time.sleep(2 ** retry_count)  # Exponential backoff
            new_response = chat_with_ai(error_message, is_edit_request=False, added_files=added_files)
            if new_response:
                return apply_creation_steps(new_response, added_files, retry_count + 1)
            else:
                return False
        else:
            print(colored(f"創建回應失敗: {str(e)}", "red"))
            logging.error(f"創建回應失敗: {str(e)}")
            print("創建回應失敗:")
            print(creation_response)
            return False
    except Exception as e:
        print(colored(f"創建回應失敗: {e}", "red"))
        logging.error(f"創建回應失敗: {e}")
        return False



def parse_edit_instructions(response):
    instructions = {}
    current_file = None
    current_instructions = []

    for line in response.split('\n'):
        if line.startswith("File: "):
            if current_file:
                instructions[current_file] = "\n".join(current_instructions)
            current_file = line[6:].strip()
            current_instructions = []
        elif line.strip() and current_file:
            current_instructions.append(line.strip())

    if current_file:
        instructions[current_file] = "\n".join(current_instructions)

    return instructions

def apply_edit_instructions(edit_instructions, original_files):
    modified_files = {}
    for file_path, content in original_files.items():
        if file_path in edit_instructions:
            instructions = edit_instructions[file_path]
            prompt = f"{APPLY_EDITS_PROMPT}\n\nOriginal File: {file_path}\nContent:\n{content}\n\nEdit Instructions:\n{instructions}\n\nUpdated File Content:"
            response = chat_with_ai(prompt, is_edit_request=True)
            if response:
                modified_files[file_path] = response.strip()
        else:
            modified_files[file_path] = content  # No changes for this file
    return modified_files

def chat_with_ai(user_message, is_edit_request=False, retry_count=0, added_files=None):
    global last_ai_response, conversation_history, MODEL
    try:
        # Include added file contents and conversation history in the user message
        if added_files:
            file_context = "Added files:\n"
            for file_path, content in added_files.items():
                file_context += f"File: {file_path}\nContent:\n{content}\n\n"
            user_message = f"{file_context}\n{user_message}"

        # Include conversation history
        if not is_edit_request:
            history = "\n".join([f"User: {msg}" if i % 2 == 0 else f"AI: {msg}" for i, msg in enumerate(conversation_history)])
            if history:
                user_message = f"{history}\nUser: {user_message}"

        # Prepare the message content based on the request type
        if is_edit_request:
            prompt = EDIT_INSTRUCTION_PROMPT if retry_count == 0 else APPLY_EDITS_PROMPT
            message_content = f"{prompt}\n\nUser request: {user_message}"
        else:
            message_content = user_message

        messages = [
            {"role": "user", "content": message_content}
        ]
        
        if is_edit_request and retry_count == 0:
            print(colored("分析文件並生成修改...", "magenta"))
            logging.info("分析文件並生成修改...")
        elif not is_edit_request:
            print(colored("軟體工程師正在思考...", "magenta"))
            logging.info("發送一般查詢到 AI.")

        response = client.chat.completions.create(
            model=MODEL,  # 在這裡使用 MODEL 變量
            messages=messages,
            max_tokens=60000  # 注意：這裡使用 max_tokens 而不是 max_completion_tokens
        )
        logging.info("Received response from AI.")
        last_ai_response = response.choices[0].message.content

        if not is_edit_request:
            # Update conversation history
            conversation_history.append(user_message)
            conversation_history.append(last_ai_response)
            if len(conversation_history) > 20:  # 10 interactions (user + AI each)
                conversation_history = conversation_history[-20:]

        return last_ai_response
    except Exception as e:
        print(colored(f"與 Stima API 通訊時發生錯誤: {e}", "red"))
        logging.error(f"與 Stima API 通訊時發生錯誤: {e}")
        return None
    


def main():
    global last_ai_response, conversation_history, client, MODEL

    parser = argparse.ArgumentParser(description="Stima 助理工程師 CLI")
    parser.add_argument("--api-key", help="請輸入您的 Stima API Key")
    parser.add_argument("--model", help="請輸入模型名稱, 預設使用 Anthropic Claude 3.5 Sonnet", default="claude-3-5-sonnet-20240620")
    args = parser.parse_args()
    
    # 定義全局 MODEL 變量
    MODEL = args.model

    # 初始化 OpenAI 客戶端
    client = OpenAI(
        base_url="https://api.stima.tech/v1",  
        api_key=args.api_key if args.api_key else "YOUR KEY"
    )

    print(colored(f"Stima engineer is ready to help you. Using model: {MODEL}", "cyan"))
    print("\nAvailable commands:")
    print(f"{colored('/edit', 'magenta'):<10} {colored('編輯文件或目錄 (跟隨路徑)', 'dark_grey')}")
    print(f"{colored('/create', 'magenta'):<10} {colored('創建文件或文件夾 (跟隨指令)', 'dark_grey')}")
    print(f"{colored('/add', 'magenta'):<10} {colored('添加文件或文件夾到上下文', 'dark_grey')}")
    print(f"{colored('/debug', 'magenta'):<10} {colored('印出最後的 AI 回應', 'dark_grey')}")
    print(f"{colored('/reset', 'magenta'):<10} {colored('重置聊天上下文並清除添加的文件', 'dark_grey')}")
    print(f"{colored('/review', 'magenta'):<10} {colored('審查代碼文件 (跟隨文件路徑)', 'dark_grey')}")
    print(f"{colored('/planning', 'magenta'):<10} {colored('生成基於您請求的詳細計劃', 'dark_grey')}")
    print(f"{colored('/quit', 'magenta'):<10} {colored('退出腳本', 'dark_grey')}")
    style = Style.from_dict({
        'prompt': 'cyan',
    })

    # Get the list of files in the current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    # Create a WordCompleter with available commands and files
    completer = WordCompleter(
        ['/edit', '/create', '/add', '/quit', '/debug', '/reset', '/review', '/planning'] + files,
        ignore_case=True
    )

    added_files = {}
    file_contents = {}

    while True:
        print()  # Add a newline before the prompt
        user_input = prompt("You: ", style=style, completer=completer).strip()

        if user_input.lower() == '/quit':
            print("Goodbye!")
            logging.info("User exited the program.")
            break

        elif user_input.lower() == '/debug':
            if last_ai_response:
                print(colored("Last AI Response:", "blue"))
                print(last_ai_response)
            else:
                print(colored("No AI response available yet.", "red"))

        elif user_input.lower() == '/reset':
            conversation_history = []
            added_files.clear()
            last_ai_response = None
            print(colored("聊天上下文和添加的文件已重置。", "green"))
            logging.info("聊天上下文和添加的文件已重置。")

        elif user_input.startswith('/add'):
            paths = user_input.split()[1:]
            if not paths:
                print(colored("請提供至少一個文件或文件夾路徑。", "red"))
                logging.warning("用戶發送 /add 而沒有文件或文件夾路徑。")
                continue

            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, added_files)
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        # Skip excluded directories
                        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'node_modules'}]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(file_path, added_files)
                else:
                    print(colored(f"錯誤: {path} 既不是文件也不是目錄。", "red"))
                    logging.error(f"{path} 既不是文件也不是目錄。")
            total_size = sum(len(content) for content in added_files.values())
            if total_size > 100000:  # Warning if total content exceeds ~100KB
                print(colored("警告: 添加的文件總大小可能會影響性能。", "red"))
                logging.warning("添加的文件總大小超過 100KB。")

        elif user_input.startswith('/edit'):
            paths = user_input.split()[1:]
            if not paths:
                print(colored("請提供至少一個文件或文件夾路徑。", "red"))
                logging.warning("用戶發送 /edit 而沒有文件或文件夾路徑。")
                continue
            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, added_files)
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        # Skip excluded directories
                        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'node_modules'}]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(file_path, added_files)
                else:
                    print(colored(f"錯誤: {path} 既不是文件也不是目錄。", "red"))
                    logging.error(f"{path} 既不是文件也不是目錄。")
            if not added_files:
                print(colored("沒有有效的文件可以編輯。", "red"))
                continue
            edit_instruction = prompt(f"Edit Instruction for all files: ", style=style).strip()

            edit_request = f"""User request: {edit_instruction}

Files to modify:
"""
            for file_path, content in added_files.items():
                edit_request += f"\nFile: {file_path}\nContent:\n{content}\n\n"

            ai_response = chat_with_ai(edit_request, is_edit_request=True, added_files=added_files)
            
            if ai_response:
                print("軟體工程師: 以下是建議的編輯指令:")
                rprint(Markdown(ai_response))

                confirm = prompt("你想要應用這些編輯指令嗎? (yes/no): ", style=style).strip().lower()
                if confirm == 'yes':
                    edit_instructions = parse_edit_instructions(ai_response)
                    modified_files = apply_edit_instructions(edit_instructions, added_files)
                    for file_path, new_content in modified_files.items():
                        apply_modifications(new_content, file_path)
                else:
                    print(colored("編輯指令未應用。", "red"))
                    logging.info("用戶選擇不應用編輯指令。")

        elif user_input.startswith('/create'):
            creation_instruction = user_input[7:].strip()  # Remove '/create' and leading/trailing whitespace
            if not creation_instruction:
                print(colored("請在 /create 後提供創建指令。", "red"))
                logging.warning("用戶發送 /create 而沒有指令。")
                continue

            create_request = f"{CREATE_SYSTEM_PROMPT}\n\nUser request: {creation_instruction}"
            ai_response = chat_with_ai(create_request, is_edit_request=False, added_files=added_files)
            
            if ai_response:
                while True:
                    print("軟體工程師: 以下是建議的創建結構:")
                    rprint(Markdown(ai_response))

                    confirm = prompt("你想要執行這些創建步驟嗎? (yes/no): ", style=style).strip().lower()
                    if confirm == 'yes':
                        success = apply_creation_steps(ai_response, added_files)
                        if success:
                            break
                        else:
                            retry = prompt("創建失敗。你想要 AI 再次嘗試嗎? (yes/no): ", style=style).strip().lower()
                            if retry != 'yes':
                                break
                            ai_response = chat_with_ai("The previous creation attempt failed. Please try again with a different approach.", is_edit_request=False, added_files=added_files)
                    else:
                        print(colored("創建步驟未執行。", "red"))
                        logging.info("用戶選擇不執行創建步驟。")
                        break

        elif user_input.startswith('/review'):
            paths = user_input.split()[1:]
            if not paths:
                print(colored("請提供至少一個文件或文件夾路徑。", "red"))
                logging.warning("用戶發送 /review 而沒有文件或文件夾路徑。")
                continue

            file_contents = {}
            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, file_contents, action='to review')
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        # Skip excluded directories
                        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'node_modules'}]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(file_path, file_contents, action='to review')
                else:
                    print(colored(f"錯誤: {path} 既不是文件也不是目錄。", "red"))
                    logging.error(f"{path} 既不是文件也不是目錄。")

            if not file_contents:
                print(colored("沒有有效的文件可以審查。", "red"))
                continue

            review_request = f"{CODE_REVIEW_PROMPT}\n\nFiles to review:\n"
            for file_path, content in file_contents.items():
                review_request += f"\nFile: {file_path}\nContent:\n{content}\n\n"

            print(colored("分析程式碼並生成審查...", "magenta"))
            ai_response = chat_with_ai(review_request, is_edit_request=False, added_files=added_files)
            
            if ai_response:
                print()
                print(colored("程式碼審查:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("提供程式碼審查給請求的文件。")

        elif user_input.startswith('/planning'):
            planning_instruction = user_input[9:].strip()  # Remove '/planning' and leading/trailing whitespace
            if not planning_instruction:
                print(colored("請在 /planning 後提供計劃請求。", "red"))
                logging.warning("用戶發送 /planning 而沒有指令。")
                continue
            planning_request = f"{PLANNING_PROMPT}\n\nUser request: {planning_instruction}"
            ai_response = chat_with_ai(planning_request, is_edit_request=False, added_files=added_files)
            if ai_response:
                print()
                print(colored("軟體工程師: 以下是你的詳細計劃:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("提供計劃回應給用戶。")
            else:
                print(colored("生成計劃回應失敗。請再試一次。", "red"))
                logging.error("AI 生成計劃回應失敗。")

        else:
            ai_response = chat_with_ai(user_input, added_files=added_files)
            if ai_response:
                print()
                print(colored("軟體工程師:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("提供 AI 回應給用戶查詢。")




if __name__ == "__main__":
    main()