# %%
#This is modified on 02012026
from dotenv import load_dotenv #
from openai import OpenAI
import json
import os
import sqlite3  # CHANGED: Switched from pyodbc to sqlite3
from pypdf import PdfReader
import gradio as gr


# %%

load_dotenv(override=True)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

# #--- SQLite Connection Logic (CHANGED) ---

# Constructing the connection for SQLite
def get_db_connection():
    # SQLite connects to a local file instead of a server
    # This creates 'chat_data.db' in the same folder if it doesn't exist
    return sqlite3.connect('chat_data.db')


def init_db():
    """Creates the table in SQLite if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    # CHANGED: SQLite syntax for table creation
    # - Used 'AUTOINCREMENT' instead of 'IDENTITY(1,1)'
    # - Used 'TEXT' instead of 'NVARCHAR'
    # - Used 'TIMESTAMP' instead of 'DATETIME'
    # - Used 'CURRENT_TIMESTAMP' instead of 'GETDATE()'
    cur.execute("""
        CREATE TABLE IF NOT EXISTS AgentPrompt_User (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            name TEXT,
            notes TEXT,
            unknown_question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

# Initialize table on startup
init_db()


# %%

# --Add agent tools

# --- Logic for Agent Tools ---

def record_user_details(email, name="Name not provided", notes="not provided"):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # SQLite uses '?' placeholders just like pyodbc, so this query syntax works for both
        cur.execute(
            "INSERT INTO AgentPrompt_User (email, name, notes) VALUES (?, ?, ?)",
            (email, name, notes)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"recorded": "ok"}
    except Exception as e:
        print(f"Database Error: {e}")
        return {"recorded": "error"}

def record_unknown_question(question):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO AgentPrompt_User (unknown_question) VALUES (?)",
            (question,)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"recorded": "ok"}
    except Exception as e:
        print(f"Database Error: {e}")
        return {"recorded": "error"}

# -Jsons format

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

#add Json tools
tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


# %%

# #Class definition for handling tool calls

class Me:
    def __init__(self):
        self.gemini = gemini
        self.name = "Hari P"
        reader = PdfReader("MyResources/LinkedinProfile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("MyResources/AboutMe.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        return f"""You are acting as {self.name}. You are answering questions on {self.name}'s website.
        
        ## Summary:
        {self.summary}
        
        ## LinkedIn Profile:
        {self.linkedin}
        
        ## INSTRUCTIONS:
        1. **Job Fit & Experience:** If the user asks about a specific role (e.g., "GCC Head", "CTO", "DATA and AI Lead"), use the experience listed above to ARGUE why you are a good fit. synthesise your leadership and technical skills to answer these questions persuasively quoting facts and numbers. Do NOT use the unknown tool for these.
        
        2. **Missing Facts:** Use the `record_unknown_question` tool ONLY if the user asks for a specific FACT that is completely absent from the text. 
           - Example: "Do you have a patent?" (If not in text -> Record it).
           - Example: "What is your favorite food?" (If not in text -> Record it).
           - Example: "What is your phone number?" (If not in text -> Record it).
        
        3. **Contact:** If the user wants to get in touch, ask for their name, followed by their email address and use `record_user_details`.
        
        4. **Tone:** Be professional, confident, and conversational as if you are acting as Global IT head of a company.
        """

    def chat(self, message, history):
            formatted_history = []
            
            # Gradio 4/5 sends history as a list of objects/dicts
            # We extract the role and content properly
            for turn in history:
                # turn is usually a dict or an object with 'role' and 'content'
                # If turn is a dict:
                role = turn.get("role")
                content = turn.get("content")
                if role and content:
                    formatted_history.append({"role": role, "content": content})

            messages = [{"role": "system", "content": self.system_prompt()}] + formatted_history + [{"role": "user", "content": message}]
            
            done = False
            while not done:
                # Correcting model name: 'gemini-2.0-flash' or 'gemini-1.5-flash'
                # (2.5 does not exist yet!)
                response = self.gemini.chat.completions.create(
                    model="gemini-2.5-flash", 
                    messages=messages, 
                    tools=tools
                )
                
                message_obj = response.choices[0].message
                
                if message_obj.tool_calls:
                    results = self.handle_tool_call(message_obj.tool_calls)
                    messages.append(message_obj)
                    messages.extend(results)
                else:
                    done = True
                    
            return response.choices[0].message.content

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat).launch()
