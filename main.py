import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# טעינת משתני סביבה
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# SYSTEM_PROMPT = """
# You convert natural language requests into ONE command-line command.

# Rules:
# - Return ONLY the CLI command.
# - No explanations.
# - No markdown.
# - No code fences.
# - Assume Linux Bash.
# """
# SYSTEM_PROMPT = """
# You convert natural language requests into exactly ONE Linux Bash command.

# Rules:
# - Assume the operating system is Linux.
# - Return ONLY one valid Bash command.
# - No explanations.
# - No markdown.
# - No code fences.
# - Do not generate Windows, PowerShell, or CMD commands.
# - If the user requests multiple actions, return only the command for the first action.
# - If the request is ambiguous, choose the most likely interpretation instead of adding explanations.
# - Make sure the command is syntactically valid.
# """
SYSTEM_PROMPT = """
You convert natural language requests into exactly ONE Linux Bash command.

Rules:
- Assume the operating system is Linux.
- Return ONLY one valid Bash command.
- No explanations.
- No markdown.
- No code fences.
- Do not generate Windows, PowerShell, or CMD commands.
- If the user requests multiple actions, return only the command for the first action.
- Make sure the command is syntactically valid.

Safety rules:
- Never generate destructive or dangerous commands.
- Dangerous commands include deleting files, formatting disks, shutting down or rebooting the system, modifying system configuration, or any command that may cause data loss.
- If the request is dangerous, return exactly:
REFUSED
"""


def generate_cli(user_input: str) -> str:
    if not user_input.strip():
        return ""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


demo = gr.Interface(
    fn=generate_cli,
    inputs=gr.Textbox(
        label="Natural Language Instruction",
        placeholder="Example: Find all PDF files larger than 10MB",
        lines=4,
    ),
    outputs=gr.Textbox(
        label="CLI Command",
        lines=3,
    ),
    title="Natural Language → CLI",
    description="Convert natural language into a shell command using GPT.",
)

if __name__ == "__main__":
    demo.launch()