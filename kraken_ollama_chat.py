"""
Kraken Ollama Chat Node for ComfyUI
Interactive prompt generation through LLM conversation
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[Kraken Ollama] Warning: ollama library not installed. Run: pip install ollama")

# Global state to maintain conversation across node executions
CHAT_STATE = {
    "conversation": [],
    "connected": False,
    "current_host": None,
    "current_model": None,
    "last_response": "",
}

class KrakenOllamaPromptChat:
    """
    Interactive Ollama chat for prompt generation in ComfyUI
    """
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.type = "KrakenOllamaPromptChat"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "host": ("STRING", {
                    "default": "http://192.168.1.93:11434",
                    "multiline": False
                }),
                "model": ("STRING", {
                    "default": "llama3.2",
                    "multiline": False
                }),
                "action": ([
                    "Connect",
                    "Send Message", 
                    "Clear Chat",
                    "Use Last Response as Prompt"
                ], {
                    "default": "Connect"
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "Create a prompt for: a futuristic cityscape at sunset"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_output", "ai_response", "conversation_log")
    FUNCTION = "chat_interface"
    CATEGORY = "🦑 Kraken/LLM"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-evaluation on every execution
        return float("nan")
    
    def get_system_prompt(self) -> str:
        """AI Art Generator system prompt"""
        return """You are an expert AI art prompt generator. Create detailed, vivid text-to-image prompts focusing purely on visual elements. Describe composition, lighting, colors, textures, camera angles, artistic style, and atmosphere. Use specific technical terms like "cinematic lighting," "bokeh," "8K resolution," "oil painting style," etc.

CRITICAL: Output ONLY the visual prompt description. No thinking process, no explanations, no meta-commentary, no quotation marks. Just pure, descriptive visual language ready to feed directly into an image generator.

Example format: "A majestic dragon perched on ancient ruins, golden hour lighting, volumetric fog, fantasy art style, highly detailed scales, dramatic composition, 4K quality"

When the user asks to refine or modify a prompt, make the changes and output the complete new prompt."""
    
    async def async_chat(self, host: str, model: str, message: str) -> str:
        """Send message to Ollama and get response"""
        if not OLLAMA_AVAILABLE:
            return "ERROR: ollama library not installed. Run: pip install ollama"
        
        try:
            self.client = AsyncClient(host=host)
            
            # Build messages with system prompt
            messages = [{"role": "system", "content": self.get_system_prompt()}]
            messages.extend(CHAT_STATE["conversation"])
            messages.append({"role": "user", "content": message})
            
            # Get response
            response = await self.client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 500}
            )
            
            ai_response = response.get("message", {}).get("content", "").strip()
            
            # Remove any thinking tags
            if "<think>" in ai_response.lower():
                ai_response = ai_response.split("</think>")[-1].strip()
            
            # Update conversation history
            CHAT_STATE["conversation"].append({"role": "user", "content": message})
            CHAT_STATE["conversation"].append({"role": "assistant", "content": ai_response})
            CHAT_STATE["last_response"] = ai_response
            
            return ai_response
        
        except Exception as e:
            error_msg = f"ERROR: {str(e)}\n\nTroubleshooting:\n1. Ensure Ollama is running on {host}\n2. Verify model '{model}' is pulled\n3. Check network/firewall settings"
            return error_msg
    
    async def async_connect(self, host: str, model: str) -> str:
        """Test connection to Ollama"""
        if not OLLAMA_AVAILABLE:
            return "ERROR: ollama library not installed"
        
        try:
            self.client = AsyncClient(host=host)
            response = await self.client.list()
            
            # Get available models
            models = []
            if hasattr(response, 'models'):
                models = [m.model for m in response.models]
            elif isinstance(response, dict):
                models = [m.get("name") or m.get("model") for m in response.get("models", [])]
            
            CHAT_STATE["connected"] = True
            CHAT_STATE["current_host"] = host
            CHAT_STATE["current_model"] = model
            
            if models:
                models_str = ", ".join(models[:5])
                if len(models) > 5:
                    models_str += f"... ({len(models)} total)"
                return f"✓ Connected to {host}\n✓ Model: {model}\n\nAvailable models: {models_str}"
            else:
                return f"✓ Connected to {host}\n✓ Model: {model}\n\n⚠ No models found. Pull a model first:\n  ollama pull {model}"
        
        except Exception as e:
            CHAT_STATE["connected"] = False
            return f"✗ Connection failed: {str(e)}"
    
    def format_conversation_log(self) -> str:
        """Format conversation for display"""
        if not CHAT_STATE["conversation"]:
            return "No conversation yet. Send a message to start."
        
        log = []
        for msg in CHAT_STATE["conversation"]:
            role = "You" if msg["role"] == "user" else "AI"
            log.append(f"{role}: {msg['content']}\n")
        
        return "\n".join(log)
    
    def chat_interface(self, host: str, model: str, action: str, user_message: str):
        """Main execution method"""
        
        prompt_output = ""
        ai_response = ""
        conversation_log = ""
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if action == "Connect":
                ai_response = loop.run_until_complete(
                    self.async_connect(host, model)
                )
                conversation_log = "Connection attempt complete. Check AI Response."
                
            elif action == "Send Message":
                if not user_message.strip():
                    ai_response = "ERROR: Please enter a message"
                else:
                    ai_response = loop.run_until_complete(
                        self.async_chat(host, model, user_message)
                    )
                    conversation_log = self.format_conversation_log()
                
            elif action == "Clear Chat":
                CHAT_STATE["conversation"].clear()
                CHAT_STATE["last_response"] = ""
                ai_response = "Chat cleared. Start a new conversation."
                conversation_log = "Conversation cleared."
                
            elif action == "Use Last Response as Prompt":
                if CHAT_STATE["last_response"]:
                    prompt_output = CHAT_STATE["last_response"]
                    ai_response = f"✓ Using last response as prompt:\n\n{prompt_output}"
                    conversation_log = self.format_conversation_log()
                else:
                    ai_response = "ERROR: No response available. Send a message first."
            
            loop.close()
        
        except Exception as e:
            ai_response = f"ERROR: {str(e)}"
            conversation_log = f"Error occurred: {str(e)}"
        
        return (prompt_output, ai_response, conversation_log)


# Node registration
NODE_CLASS_MAPPINGS = {
    "KrakenOllamaPromptChat": KrakenOllamaPromptChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenOllamaPromptChat": "🦑 Kraken Ollama Prompt Chat",
}