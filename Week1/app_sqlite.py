#updated on Jan 31
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import sqlite3
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

# --- SQLite Connection Logic ---
def get_db_connection():
    conn = sqlite3.connect('chat_data.db')
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
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
    conn.close()

init_db()

# --- Agent Tools ---
def record_user_details(email, name="Name not provided", notes="not provided"):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO AgentPrompt_User (email, name, notes) VALUES (?, ?, ?)", (email, name, notes))
        conn.commit()
        conn.close()
        return {"recorded": "ok"}
    except Exception as e:
        return {"recorded": "error", "details": str(e)}

def record_unknown_question(question):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO AgentPrompt_User (unknown_question) VALUES (?)", (question,))
        conn.commit()
        conn.close()
        return {"recorded": "ok"}
    except Exception as e:
        return {"recorded": "error", "details": str(e)}

# --- Tool Definitions ---
record_user_details_json = {
    "name": "record_user_details",
    "description": "Record user contact info",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["email"]
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record unanswered questions",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"]
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]

# --- Agent Class ---
class Me:
    def __init__(self):
        self.gemini = gemini
        self.name = "Hari P"
        
        try:
            reader = PdfReader("MyResources/LinkedinProfile.pdf")
            self.linkedin = ""
            for page in reader.pages:
                text = page.extract_text()
                if text: self.linkedin += text
        except: self.linkedin = "Profile not available."

        try:
            with open("MyResources/AboutMe.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except: self.summary = "Summary not available."

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            res = tool(**args) if tool else {}
            results.append({"role": "tool", "content": json.dumps(res), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        return f"""You are acting as {self.name}. You are answering questions on {self.name}'s website.
        
        ## Summary:
        {self.summary}
        
        ## LinkedIn Profile:
        {self.linkedin}
        
        If you don't know the answer, use record_unknown_question. 
        If the user wants to get in touch, ask for email and use record_user_details.
        """

    def chat(self, message, history):
        formatted_history = []
        
        # FIX: Manual conversion for older Gradio versions
        # Older Gradio sends history as [[user, bot], [user, bot]]
        for user_msg, bot_msg in history:
            formatted_history.append({"role": "user", "content": user_msg})
            formatted_history.append({"role": "assistant", "content": bot_msg})

        messages = [{"role": "system", "content": self.system_prompt()}] + formatted_history + [{"role": "user", "content": message}]
        
        done = False
        while not done:
            response = self.gemini.chat.completions.create(
                model="gemini-1.5-flash", 
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
    # FIX: Removed type="messages" to support your older Gradio version
    gr.ChatInterface(me.chat).launch()